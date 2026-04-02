/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

#include <stddef.h>
#include <stdlib.h>

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Memory allocation parameter used by ArenaPlanner.
// Clients (such as delegates) might look at this to ensure interop between
// TFLite memory & hardware buffers.
// NOTE: This only holds for tensors allocated on the arena.
constexpr int kDefaultTensorAlignment = 64;

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'ndims' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(int ndims, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);

// The prefix of a validation subgraph name.
// WARNING: This is an experimental API and subject to change.
constexpr char kValidationSubgraphNamePrefix[] = "VALIDATION:";

// Checks whether the prefix of the subgraph name indicates the subgraph is a
// validation subgraph.
bool IsValidationSubgraph(const char* name);

// Multiply two sizes and return true if overflow occurred;
// This is based off tensorflow/overflow.h but is simpler as we already
// have unsigned numbers. It is also generalized to work where sizeof(size_t)
// is not 8.
TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product);

// Returns whether the TfLiteTensor is a resource or variant tensor.
inline bool IsResourceOrVariant(const TfLiteTensor* tensor) {
  return tensor->type == kTfLiteResource || tensor->type == kTfLiteVariant;
}

// Compute the number of bytes required to represent a tensor with dimensions
// specified by the array dims (of length dims_size). Returns the status code
// and bytes.
TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                           size_t* bytes, TfLiteContext* context);

// `unique_ptr` wrapper for `TfLiteTensor`s.
struct TfLiteTensorDeleter {
  void operator()(TfLiteTensor* t) {
    if (t) {
      TfLiteTensorFree(t);
    }
    free(t);
  }
};

using TensorUniquePtr = std::unique_ptr<TfLiteTensor, TfLiteTensorDeleter>;
TensorUniquePtr BuildTfLiteTensor();
TensorUniquePtr BuildTfLiteTensor(TfLiteType type, const std::vector<int>& dims,
                                  TfLiteAllocationType allocation_type);
TensorUniquePtr BuildTfLiteTensor(TfLiteType type, IntArrayUniquePtr dims,
                                  TfLiteAllocationType allocation_type);

int GetBuiltinDataSize(BuiltinOperator op);

// Wraps an integer value and allows checking if standard arithmetic operations
// used to generate that value have over/underflowed.
template <class T>
class CheckedInt {
 public:
  using type = T;

  static_assert(std::is_integral_v<T>, "T must but an integral value.");

  CheckedInt() = default;
  CheckedInt(const CheckedInt&) = default;
  CheckedInt(CheckedInt&&) = default;
  CheckedInt& operator=(const CheckedInt&) = default;
  CheckedInt& operator=(CheckedInt&&) = default;

  // NOLINTNEXTLINE(*-explicit-constructor): we want implicit conversion.
  CheckedInt(T val) : value_(val), overflow_(false) {}

  template <class U>
  explicit CheckedInt(U val) : value_(static_cast<T>(val)) {
    if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {
      overflow_ = val < std::numeric_limits<T>::lowest() ||
                  val > std::numeric_limits<T>::max();
    } else if constexpr (!std::is_signed_v<U> && std::is_signed_v<T>) {
      overflow_ = val > static_cast<U>(std::numeric_limits<T>::max());
    } else {
      overflow_ = val < 0 || static_cast<std::make_unsigned_t<U>>(val) >
                                 std::numeric_limits<T>::max();
    }
  }

  template <class U>
  CheckedInt& operator=(const CheckedInt<U> other) {
    *this = CheckedInt<T>(other.value_);
    overflow_ |= other.overflow_;
    return *this;
  }

  T Value() const noexcept { return value_; }

  bool Overflow() const noexcept { return overflow_; }

  TfLiteStatus Status() const noexcept {
    return overflow_ ? kTfLiteError : kTfLiteOk;
  }

  template <class U>
  CheckedInt& operator+=(const CheckedInt<U>& b) noexcept {
    return *this = *this + b;
  }

  template <class U>
  CheckedInt& operator-=(const CheckedInt<U>& b) noexcept {
    return *this = *this - b;
  }

  template <class U>
  CheckedInt& operator*=(const CheckedInt<U>& b) noexcept {
    return *this = *this * b;
  }

  template <class U>
  CheckedInt& operator/=(const CheckedInt<U>& b) noexcept {
    return *this = *this / b;
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator+=(U b) noexcept {
    return *this += CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator-=(U b) noexcept {
    return *this -= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator*=(U b) noexcept {
    return *this *= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator/=(U b) noexcept {
    return *this /= CheckedInt<U>(b);
  }

 private:
  // Helper constructor for operators.
  CheckedInt(T val, bool overflow) : value_(val), overflow_(overflow) {}

  template <class U>
  friend class CheckedInt;

  template <class U>
  using CommonType = CheckedInt<std::common_type_t<T, U>>;

  template <class U>
  friend CommonType<U> operator+(const CheckedInt<T>& a,
                                 const CheckedInt<U>& b) noexcept {
    CommonType<U> res;
#if defined(__GNUC__) || defined(__clang__)
    res.overflow_ = __builtin_add_overflow(a.value_, b.value_, &res.value_) ||
                    a.overflow_ || b.overflow_;
#else
    using limits = std::numeric_limits<typename CommonType<U>::type>;
    res.overflow_ = a.overflow_ || b.overflow_ ||
                    (b.value_ > 0 && a.value_ >= limits::max() - b.value_) ||
                    (b.value_ < 0 && a.value_ <= limits::lowest() - b.value_);
    res.value_ = a.value_ + b.value_;
#endif
    return res;
  }

  template <class U>
  friend CommonType<U> operator-(const CheckedInt<T>& a,
                                 const CheckedInt<U>& b) noexcept {
    CommonType<U> res;
#if defined(__GNUC__) || defined(__clang__)
    res.overflow_ = __builtin_sub_overflow(a.value_, b.value_, &res.value_) ||
                    a.overflow_ || b.overflow_;
#else
    using limits = std::numeric_limits<typename CommonType<U>::type>;
    res.overflow_ = a.overflow_ || b.overflow_ ||
                    (b.value_ > 0 && a.value_ < limits::lowest() + b.value_) ||
                    (b.value_ < 0 && a.value_ > limits::max() + b.value_);
    res.value_ = a.value_ - b.value_;
#endif
    return res;
  }

  template <class U>
  friend CommonType<U> operator*(const CheckedInt<T>& a,
                                 const CheckedInt<U>& b) noexcept {
    CommonType<U> res;
#if defined(__GNUC__) || defined(__clang__)
    res.overflow_ = __builtin_mul_overflow(a.value_, b.value_, &res.value_) ||
                    a.overflow_ || b.overflow_;
#else
    using limits = std::numeric_limits<typename CommonType<U>::type>;
    res.overflow_ =
        a.overflow_ || b.overflow_ ||
        (a.value_ > 0 && b.value_ > 0 && a.value_ > limits::max() / b.value_) ||
        (a.value_ < 0 && b.value_ < 0 && a.value_ < limits::max() / b.value_) ||
        (a.value_ > 0 && b.value_ < 0 &&
         b.value_ < limits::lowest() / a.value_) ||
        (a.value_ < 0 && b.value_ > 0 &&
         a.value_ < limits::lowest() / b.value_);
    res.value_ = a.value_ * b.value_;
#endif
    return res;
  }

  template <class U>
  friend CommonType<U> operator/(const CheckedInt<T>& a,
                                 const CheckedInt<U>& b) noexcept {
    using limits = std::numeric_limits<typename CommonType<U>::type>;
    if constexpr (std::is_signed_v<T> && std::is_signed_v<U>) {
      if (a.value_ == limits::lowest() && b.value_ == -1) {
        return {/*value=*/limits::max(), /*overflow=*/true};
      }
    }
    return {/*value=*/b.value_ != 0 ? a.value_ / b.value_ : limits::max(),
            /*overflow=*/b.value_ == 0 || a.overflow_ || b.overflow_};
  }

  template <class U>
  friend bool operator==(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a == b.Value();
  }

  template <class U>
  friend bool operator!=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return !(a == b);
  }

  template <class U>
  friend bool operator<(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return a < b.Value();
  }

  template <class U>
  friend bool operator<=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a <= b.Value();
  }

  template <class U>
  friend bool operator>(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return b < a.Value();
  }

  template <class U>
  friend bool operator>=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return b <= a.Value();
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(OP)                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(const CheckedInt<T>& a, U b) noexcept {        \
    return a OP CheckedInt<U>(b);                                        \
  }                                                                      \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(U a, const CheckedInt<T>& b) noexcept {        \
    return CheckedInt<U>(a) OP b;                                        \
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(OP)                            \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(const CheckedInt<T>& a, U b) noexcept {             \
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {               \
      return a.Value() OP b;                                                  \
    } else if constexpr (std::is_signed_v<T>) {                               \
      return a.Value() >= 0 ? static_cast<std::make_unsigned_t<T>>(a.Value()) \
                                  OP b                                        \
                            : (0 OP 1);                                       \
    } else {                                                                  \
      return b >= 0 ? a.Value() OP static_cast<std::make_unsigned_t<U>>(b)    \
                    : (1 OP 0);                                               \
    }                                                                         \
  }                                                                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(U a, const CheckedInt<T>& b) noexcept {             \
    if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {               \
      return a OP b.Value();                                                  \
    } else if constexpr (std::is_signed_v<U>) {                               \
      return a >= 0 ? static_cast<std::make_unsigned_t<U>>(a) OP b.Value()    \
                    : (0 OP 1);                                               \
    } else {                                                                  \
      return b.Value() >= 0                                                   \
                 ? a OP static_cast<std::make_unsigned_t<T>>(b.Value())       \
                 : (1 OP 0);                                                  \
    }                                                                         \
  }

  // NOLINTBEGIN(whitespace/operators)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(+)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(-)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(*)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(/)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(==)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(!=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>=)
  // NOLINTEND(whitespace/operators)
  //
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_OP
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP

 private:
  T value_{};
  bool overflow_ = false;
};

template <class T>
CheckedInt(T) -> CheckedInt<T>;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
