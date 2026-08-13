/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/cuda/cudart_kernel_registry.h"

#include <elf.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace stream_executor::cuda {
namespace {

// Standard uncompressed CUDA Fatbinary container magic number (NVCC).
constexpr uint32_t kFatbinMagicUncompressed = 0xba55d10a;

// Compressed CUDA Fatbinary container magic number (CUDA 11+ / 12+ NVCC).
constexpr uint32_t kFatbinMagicCompressed = 0xba55ed50;

// Legacy/alternative CUDA Fatbinary payload header magic number.
constexpr uint32_t kFatbinMagicLegacy = 0x00101001;

// NVIDIA CUDA ELF machine type (EM_CUDA). Not defined by every <elf.h>.
constexpr uint16_t kElfMachineCuda = 190;

// Encoding of the SM architecture number in an NVIDIA CUBIN ELF's `e_flags`.
// The SM number equals major*10 + minor (e.g. sm_90 -> 90 -> 0x5a). Prior to
// Blackwell the number is stored in the low 8 bits; starting with Blackwell
// (sm_100+) it is stored in bits 8-15 instead. See llvm/BinaryFormat/ELF.h
// (EF_CUDA_SM, EF_CUDA_SM_MASK, EF_CUDA_SM_OFFSET, EF_CUDA_SM* enum).
constexpr uint32_t kElfCudaSmMask = 0xff;
constexpr uint32_t kElfCudaSmOffset = 8;
// EF_CUDA_SM100: the first Blackwell SM number, above which the SM number is
// stored in bits 8-15 rather than the low 8 bits.
constexpr int kElfCudaSmBlackwell = 0x64;

// CUDA fatbin wrapper structure passed to __cudaRegisterFatBinary.
struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void* data;
  void* filename_or_fatbins;
};

struct FatHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t fat_size;
};

// Reads a little-endian uint32_t from `data` at `offset` (bounds must be
// checked by the caller).
uint32_t ReadU32LE(absl::Span<const uint8_t> data, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

// Helper function to parse CUDA Fatbinary containers or raw ELF CUBINs from
// a FatbinWrapper pointer and return an absl::Span pointing to the binary
// payload.
std::optional<absl::Span<const uint8_t>> ParseFatBinaryOrElf(
    const void* fat_cubin) {
  if (fat_cubin == nullptr) {
    return std::nullopt;
  }

  const auto* wrapper = static_cast<const FatbinWrapper*>(fat_cubin);
  if (wrapper->data == nullptr) {
    return std::nullopt;
  }

  const uint8_t* data_bytes = static_cast<const uint8_t*>(wrapper->data);
  size_t total_size = 0;
  const auto* header = reinterpret_cast<const FatHeader*>(data_bytes);
  if (header->magic == kFatbinMagicUncompressed ||
      header->magic == kFatbinMagicCompressed) {
    total_size = static_cast<size_t>(header->header_size) +
                 static_cast<size_t>(header->fat_size);
  } else if (header->magic == kFatbinMagicLegacy) {
    total_size =
        static_cast<size_t>(*reinterpret_cast<const uint64_t*>(data_bytes + 8));
  } else if (data_bytes[0] == 0x7f && data_bytes[1] == 'E' &&
             data_bytes[2] == 'L' && data_bytes[3] == 'F') {
    const auto* elf_header = reinterpret_cast<const Elf64_Ehdr*>(data_bytes);
    total_size = static_cast<size_t>(
        elf_header->e_shoff +
        (static_cast<uint64_t>(elf_header->e_shnum) * elf_header->e_shentsize));
  } else {
    return std::nullopt;
  }

  return absl::Span<const uint8_t>(data_bytes, total_size);
}

// Returns true if `data` begins with a little-endian 64-bit NVIDIA CUDA ELF
// header (a CUBIN).
bool IsCudaElf(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(Elf64_Ehdr)) {
    return false;
  }
  if (!(data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' &&
        data[3] == 'F')) {
    return false;
  }
  const auto* header = reinterpret_cast<const Elf64_Ehdr*>(data.data());
  return header->e_ident[EI_CLASS] == ELFCLASS64 &&
         header->e_ident[EI_DATA] == ELFDATA2LSB &&
         header->e_machine == kElfMachineCuda;
}

// Returns the total size in bytes of the CUBIN ELF that begins at `data`
// (assuming the section header table is the last structure in the image, which
// holds for nvcc-produced cubins), or nullopt if the header is inconsistent.
std::optional<size_t> CudaElfSize(absl::Span<const uint8_t> data) {
  if (data.size() < sizeof(Elf64_Ehdr)) {
    return std::nullopt;
  }
  const auto* header = reinterpret_cast<const Elf64_Ehdr*>(data.data());
  size_t sections_end =
      static_cast<size_t>(header->e_shoff) +
      static_cast<size_t>(header->e_shnum) * header->e_shentsize;
  size_t segments_end =
      static_cast<size_t>(header->e_phoff) +
      static_cast<size_t>(header->e_phnum) * header->e_phentsize;
  size_t size = std::max(sections_end, segments_end);
  if (size == 0 || size > data.size()) {
    return std::nullopt;
  }
  return size;
}

// Returns the SM architecture number (major*10 + minor) encoded in a CUBIN
// ELF header's `e_flags`, handling both the pre-Blackwell (low 8 bits) and
// Blackwell+ (bits 8-15) encodings.
int CudaElfSmArch(const Elf64_Ehdr* header) {
  const int blackwell_sm =
      static_cast<int>((header->e_flags >> kElfCudaSmOffset) & kElfCudaSmMask);
  if (blackwell_sm >= kElfCudaSmBlackwell) {
    return blackwell_sm;
  }
  return static_cast<int>(header->e_flags & kElfCudaSmMask);
}

// Walks the (uncompressed) fatbin `fatbin` and returns the CUBIN ELF for the
// architecture matching `cc`, as a sub-span of `fatbin`. Returns nullopt if no
// matching CUBIN is present.
//
// Rather than parsing NVIDIA's undocumented per-entry fatbin headers, this
// scans for embedded, well-formed NVIDIA CUDA ELF images and matches on the
// architecture encoded in each image's ELF header. This is robust to fatbin
// container layout changes. It relies on the images being uncompressed, which
// embeddable_cuda_library guarantees.
std::optional<absl::Span<const uint8_t>> FindCubinForArch(
    absl::Span<const uint8_t> fatbin, const CudaComputeCapability& cc) {
  const int wanted_sm = cc.major * 10 + cc.minor;
  size_t pos = 0;
  while (pos + sizeof(Elf64_Ehdr) <= fatbin.size()) {
    const uint8_t* base = fatbin.data();
    const void* hit = std::memchr(base + pos, 0x7f, fatbin.size() - pos);
    if (hit == nullptr) {
      break;
    }
    size_t offset = static_cast<const uint8_t*>(hit) - base;
    absl::Span<const uint8_t> candidate = fatbin.subspan(offset);
    if (IsCudaElf(candidate)) {
      std::optional<size_t> size = CudaElfSize(candidate);
      if (size.has_value()) {
        const auto* header =
            reinterpret_cast<const Elf64_Ehdr*>(candidate.data());
        if (CudaElfSmArch(header) == wanted_sm) {
          return fatbin.subspan(offset, *size);
        }
        // Skip past this CUBIN's body to continue searching for other images.
        pos = offset + *size;
        continue;
      }
    }
    pos = offset + 1;
  }
  return std::nullopt;
}

// Parses NVIDIA's `.nv.info.<kernel>` TLV blob, filling in the fields of
// `attrs` that it encodes.
//
// The `.nv.info` format is undocumented and reverse-engineered. Each record is
// a (format, attribute) byte pair followed by a value whose encoding depends on
// the format:
//   EIFMT_NVAL (0x01): no value.
//   EIFMT_BVAL (0x02): a single inline byte.
//   EIFMT_HVAL (0x03): a two-byte inline value.
//   EIFMT_SVAL (0x04): a two-byte little-endian length, then that many bytes.
// Records are padded so that each subsequent record starts on a 4-byte
// boundary. This padding is essential: without it the walker desyncs on
// NVAL/BVAL/HVAL records (verified against real cubins with `cuobjdump -elf`).
void ParseNvInfo(absl::Span<const uint8_t> info,
                 CudaKernelFuncAttributes* attrs) {
  constexpr uint8_t kEifmtNval = 0x01;
  constexpr uint8_t kEifmtBval = 0x02;
  constexpr uint8_t kEifmtHval = 0x03;
  constexpr uint8_t kEifmtSval = 0x04;
  // EIATTR_MAX_THREADS: launch-bound block dimensions (3 x uint32: x, y, z).
  constexpr uint8_t kAttrMaxThreads = 0x05;
  // EIATTR_FRAME_SIZE: per-thread stack frame size (uint32).
  constexpr uint8_t kAttrFrameSize = 0x11;

  size_t pos = 0;
  while (pos + 2 <= info.size()) {
    const uint8_t format = info[pos];
    const uint8_t attribute = info[pos + 1];
    pos += 2;

    if (format == kEifmtNval) {
      // No value.
    } else if (format == kEifmtBval) {
      pos += 1;
    } else if (format == kEifmtHval) {
      pos += 2;
    } else if (format == kEifmtSval) {
      if (pos + 2 > info.size()) {
        break;
      }
      const uint16_t length = static_cast<uint16_t>(info[pos]) |
                              (static_cast<uint16_t>(info[pos + 1]) << 8);
      pos += 2;
      if (pos + length > info.size()) {
        break;
      }
      absl::Span<const uint8_t> value = info.subspan(pos, length);
      if (attribute == kAttrMaxThreads && length >= 12) {
        const uint32_t x = ReadU32LE(value, 0);
        const uint32_t y = ReadU32LE(value, 4);
        const uint32_t z = ReadU32LE(value, 8);
        attrs->max_threads_per_block = static_cast<int>(x * y * z);
      } else if (attribute == kAttrFrameSize && length >= 4) {
        attrs->local_size_bytes = ReadU32LE(value, 0);
      }
      pos += length;
    } else {
      // Unknown record format: we can no longer reliably advance.
      break;
    }

    // Each record is padded so the next one starts on a 4-byte boundary.
    pos = (pos + 3) & ~static_cast<size_t>(3);
  }
}

// Returns the null-terminated name at byte `offset` within the section-name
// string table `strtab`.
absl::string_view NameAt(absl::Span<const uint8_t> strtab, uint32_t offset) {
  if (offset >= strtab.size()) {
    return {};
  }
  const char* start = reinterpret_cast<const char*>(strtab.data()) + offset;
  size_t max_length = strtab.size() - offset;
  size_t length = 0;
  while (length < max_length && start[length] != '\0') {
    ++length;
  }
  return absl::string_view(start, length);
}

// Scans the generic (non per-kernel) `.nv.info` section for the register count
// of the kernel whose symbol-table index is `symbol_index`, or nullopt if
// absent.
//
// EIATTR_REGCOUNT (0x2f) is an SVAL record whose 8-byte value is two uint32s:
// the kernel's symbol-table index followed by its register count. Newer
// architectures (sm_90+) encode the register count only here (the `.text`
// section's sh_info high byte is 0), whereas older ones (<= sm_80) additionally
// pack it into the high byte of the `.text` section's sh_info. Reading it from
// here works uniformly across architectures.
std::optional<int> ParseNvInfoRegCount(absl::Span<const uint8_t> info,
                                       uint32_t symbol_index) {
  constexpr uint8_t kEifmtNval = 0x01;
  constexpr uint8_t kEifmtBval = 0x02;
  constexpr uint8_t kEifmtHval = 0x03;
  constexpr uint8_t kEifmtSval = 0x04;
  constexpr uint8_t kAttrRegCount = 0x2f;

  size_t pos = 0;
  while (pos + 2 <= info.size()) {
    const uint8_t format = info[pos];
    const uint8_t attribute = info[pos + 1];
    pos += 2;

    if (format == kEifmtNval) {
      // No value.
    } else if (format == kEifmtBval) {
      pos += 1;
    } else if (format == kEifmtHval) {
      pos += 2;
    } else if (format == kEifmtSval) {
      if (pos + 2 > info.size()) {
        break;
      }
      const uint16_t length = static_cast<uint16_t>(info[pos]) |
                              (static_cast<uint16_t>(info[pos + 1]) << 8);
      pos += 2;
      if (pos + length > info.size()) {
        break;
      }
      absl::Span<const uint8_t> value = info.subspan(pos, length);
      if (attribute == kAttrRegCount && length >= 8 &&
          ReadU32LE(value, 0) == symbol_index) {
        return static_cast<int>(ReadU32LE(value, 4));
      }
      pos += length;
    } else {
      break;
    }

    // Each record is padded so the next one starts on a 4-byte boundary.
    pos = (pos + 3) & ~static_cast<size_t>(3);
  }
  return std::nullopt;
}

// Extracts function attributes for `mangled_name` from the CUBIN ELF `cubin`.
absl::StatusOr<CudaKernelFuncAttributes> ParseFuncAttributesFromCubin(
    absl::Span<const uint8_t> cubin, absl::string_view mangled_name,
    const CudaComputeCapability& cc) {
  CudaKernelFuncAttributes attrs;
  attrs.compute_capability = cc;

  if (cubin.size() < sizeof(Elf64_Ehdr)) {
    return absl::InvalidArgumentError("CUBIN is too small to be an ELF file");
  }
  const auto* header = reinterpret_cast<const Elf64_Ehdr*>(cubin.data());

  const size_t section_table_end =
      static_cast<size_t>(header->e_shoff) +
      static_cast<size_t>(header->e_shnum) * sizeof(Elf64_Shdr);
  if (header->e_shentsize != sizeof(Elf64_Shdr) ||
      section_table_end > cubin.size() ||
      header->e_shstrndx >= header->e_shnum) {
    return absl::InvalidArgumentError("CUBIN has an invalid section table");
  }
  const auto* sections =
      reinterpret_cast<const Elf64_Shdr*>(cubin.data() + header->e_shoff);

  // Section-name string table.
  const Elf64_Shdr& strtab_section = sections[header->e_shstrndx];
  if (static_cast<size_t>(strtab_section.sh_offset) + strtab_section.sh_size >
      cubin.size()) {
    return absl::InvalidArgumentError("CUBIN string table is out of bounds");
  }
  absl::Span<const uint8_t> strtab =
      cubin.subspan(strtab_section.sh_offset, strtab_section.sh_size);

  const std::string text_name = absl::StrCat(".text.", mangled_name);
  const std::string shared_name = absl::StrCat(".nv.shared.", mangled_name);
  const std::string info_name = absl::StrCat(".nv.info.", mangled_name);
  const std::string kernel_suffix = absl::StrCat(".", mangled_name);

  // Populated from the kernel's `.text` section; used to look up the register
  // count in the generic `.nv.info` section after the loop.
  bool found_text = false;
  uint32_t text_sh_info = 0;
  absl::Span<const uint8_t> generic_nv_info;

  for (int i = 0; i < header->e_shnum; ++i) {
    const Elf64_Shdr& section = sections[i];
    absl::string_view name = NameAt(strtab, section.sh_name);
    if (name == text_name) {
      found_text = true;
      text_sh_info = section.sh_info;
      // Fallback register count: on architectures <= sm_80 it is packed into
      // the high byte of sh_info. The generic `.nv.info` lookup below overrides
      // this when present (and is the only source on sm_90+).
      attrs.num_regs = static_cast<int>((section.sh_info >> 24) & 0xff);
      // The number of named barriers is packed into bits 20-24 of sh_flags
      // (best-effort).
      attrs.num_barriers = static_cast<int>((section.sh_flags >> 20) & 0x1f);
    } else if (name == ".nv.info") {
      if (static_cast<size_t>(section.sh_offset) + section.sh_size <=
          cubin.size()) {
        generic_nv_info = cubin.subspan(section.sh_offset, section.sh_size);
      }
    } else if (name == shared_name) {
      attrs.static_shared_size_bytes = section.sh_size;
    } else if (name == info_name) {
      if (static_cast<size_t>(section.sh_offset) + section.sh_size <=
          cubin.size()) {
        ParseNvInfo(cubin.subspan(section.sh_offset, section.sh_size), &attrs);
      }
    } else if (absl::StartsWith(name, ".nv.constant") &&
               !absl::StartsWith(name, ".nv.constant0.") &&
               absl::EndsWith(name, kernel_suffix)) {
      // Best-effort: sum the sizes of the kernel's constant banks, excluding
      // bank 0 which holds kernel parameters / ABI data rather than user
      // `__constant__` memory.
      attrs.const_size_bytes += section.sh_size;
    }
  }

  // Prefer the register count from the generic `.nv.info` section, keyed by the
  // kernel's symbol-table index (the low 24 bits of the `.text` section's
  // sh_info). This is the only reliable source on sm_90+.
  if (found_text && !generic_nv_info.empty()) {
    const uint32_t symbol_index = text_sh_info & 0x00ffffff;
    std::optional<int> reg_count =
        ParseNvInfoRegCount(generic_nv_info, symbol_index);
    if (reg_count.has_value()) {
      attrs.num_regs = *reg_count;
    }
  }

  return attrs;
}

// Captured fatbin blob together with the mangled kernel name.
struct FatbinAndName {
  absl::Span<const uint8_t> fatbin;
  absl::string_view name;
};

class KernelRegistry {
 public:
  void RegisterFatBinary(void** handle, const void* fat_cubin);
  void RegisterCudaRuntimeKernel(void** handle, const void* host_fun,
                                 absl::string_view name);
  absl::StatusOr<FatbinAndName> FindFatbinAndName(const void* host_fun) const;

 private:
  struct RegisteredKernel {
    void** fatbin_handle;
    std::string name;
  };

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<void**, absl::Span<const uint8_t>> fatbin_map_
      ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<const void*, RegisteredKernel> kernel_map_
      ABSL_GUARDED_BY(mutex_);
};

void KernelRegistry::RegisterFatBinary(void** handle, const void* fat_cubin) {
  if (handle == nullptr || fat_cubin == nullptr) {
    return;
  }
  auto span_opt = ParseFatBinaryOrElf(fat_cubin);
  if (!span_opt.has_value()) {
    return;
  }
  absl::MutexLock lock(mutex_);
  fatbin_map_[handle] = *span_opt;
}

void KernelRegistry::RegisterCudaRuntimeKernel(void** handle,
                                               const void* host_fun,
                                               absl::string_view name) {
  if (host_fun == nullptr) {
    return;
  }
  absl::MutexLock lock(mutex_);
  kernel_map_[host_fun] = {handle, std::string(name)};
}

absl::StatusOr<FatbinAndName> KernelRegistry::FindFatbinAndName(
    const void* host_fun) const {
  if (host_fun == nullptr) {
    return absl::InvalidArgumentError("Host function pointer cannot be null");
  }
  absl::MutexLock lock(mutex_);

  auto k_it = kernel_map_.find(host_fun);
  if (k_it == kernel_map_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Kernel with host function pointer %p not found in CUDA runtime kernel "
        "registry. Did you use the embeddable_cuda_library Bazel macro?",
        host_fun));
  }

  void** fat_handle = k_it->second.fatbin_handle;
  auto f_it = fatbin_map_.find(fat_handle);
  if (f_it == fatbin_map_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Fatbinary for kernel '%s' (handle %p) not found in CUDA runtime "
        "kernel registry",
        k_it->second.name, static_cast<const void*>(fat_handle)));
  }

  return FatbinAndName{f_it->second, k_it->second.name};
}

KernelRegistry& GetKernelRegistry() {
  static absl::NoDestructor<KernelRegistry> registry;
  return *registry;
}

}  // namespace

absl::StatusOr<CudaRuntimeKernel> FindCudaRuntimeKernel(
    const void* host_fun, const CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(FatbinAndName fatbin_and_name,
                   GetKernelRegistry().FindFatbinAndName(host_fun));
  std::optional<absl::Span<const uint8_t>> cubin =
      FindCubinForArch(fatbin_and_name.fatbin, compute_capability);
  if (!cubin.has_value()) {
    return absl::NotFoundError(absl::StrFormat(
        "No CUBIN for sm_%d%d found for kernel '%s'. The fatbin may be "
        "PTX-only or compressed; build the kernel with embeddable_cuda_library "
        "so its fatbin is emitted uncompressed.",
        compute_capability.major, compute_capability.minor,
        fatbin_and_name.name));
  }
  return CudaRuntimeKernel{*cubin, fatbin_and_name.name};
}

absl::StatusOr<CudaKernelFuncAttributes> FindCudaRuntimeKernelFuncAttributes(
    const void* host_fun, const CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(FatbinAndName fatbin_and_name,
                   GetKernelRegistry().FindFatbinAndName(host_fun));
  std::optional<absl::Span<const uint8_t>> cubin =
      FindCubinForArch(fatbin_and_name.fatbin, compute_capability);
  if (!cubin.has_value()) {
    return absl::NotFoundError(absl::StrFormat(
        "No CUBIN for sm_%d%d found for kernel '%s'. The fatbin may be "
        "PTX-only or compressed; build the kernel with embeddable_cuda_library "
        "so its fatbin is emitted uncompressed.",
        compute_capability.major, compute_capability.minor,
        fatbin_and_name.name));
  }
  return ParseFuncAttributesFromCubin(*cubin, fatbin_and_name.name,
                                      compute_capability);
}

}  // namespace stream_executor::cuda

extern "C" {

// Declarations of symbols we want to wrap.
void** __cudaRegisterFatBinary(void* fat_cubin);
void __cudaRegisterFunction(void** fat_cubin_handle, const char* host_fun,
                            char* device_fun, const char* device_name,
                            int thread_limit, void* tid, void* bid, void* b_dim,
                            void* g_dim, int* w_size);

// When this library is linked as a shared object (.so), the linker might omit
// wrapping of __cudaRegisterFatBinary/__cudaRegisterFunction if there are no
// references to them in the library. This dummy function forces references to
// these symbols, ensuring the linker applies the --wrap flags and resolves
// the corresponding __real_ symbols. We call this from the wrapper to prevent
// the compiler/linker from optimizing it away.
uintptr_t dummy_use_to_force_wrap() {
  volatile uintptr_t p1 = reinterpret_cast<uintptr_t>(&__cudaRegisterFatBinary);
  volatile uintptr_t p2 = reinterpret_cast<uintptr_t>(&__cudaRegisterFunction);
  return p1 + p2;
}

// Declaration of real symbols provided by CUDA runtime.
void** __real___cudaRegisterFatBinary(void* fat_cubin);
void __real___cudaRegisterFunction(void** fat_cubin_handle,
                                   const char* host_fun, char* device_fun,
                                   const char* device_name, int thread_limit,
                                   void* tid, void* bid, void* b_dim,
                                   void* g_dim, int* w_size);

// Linker wrapper intercepting calls to __cudaRegisterFatBinary.
void** __wrap___cudaRegisterFatBinary(void* fat_cubin) {
  uintptr_t dummy = dummy_use_to_force_wrap();
  (void)dummy;
  void** handle = __real___cudaRegisterFatBinary(fat_cubin);
  ::stream_executor::cuda::GetKernelRegistry().RegisterFatBinary(handle,
                                                                 fat_cubin);
  return handle;
}

// Linker wrapper intercepting calls to __cudaRegisterFunction.
void __wrap___cudaRegisterFunction(void** fat_cubin_handle,
                                   const char* host_fun, char* device_fun,
                                   const char* device_name, int thread_limit,
                                   void* tid, void* bid, void* b_dim,
                                   void* g_dim, int* w_size) {
  ::stream_executor::cuda::GetKernelRegistry().RegisterCudaRuntimeKernel(
      fat_cubin_handle, reinterpret_cast<const void*>(host_fun),
      device_name != nullptr ? device_name : "");

  __real___cudaRegisterFunction(fat_cubin_handle, host_fun, device_fun,
                                device_name, thread_limit, tid, bid, b_dim,
                                g_dim, w_size);
}

}  // extern "C"
