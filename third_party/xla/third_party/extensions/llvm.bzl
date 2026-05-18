"""Module extension for llvm."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("//third_party/net_zstd:workspace.bzl", net_zstd = "repo")

def _llvm_zlib_compat_impl(repository_ctx):
    repository_ctx.file("BUILD", """
alias(
    name = "zlib-ng",
    actual = "@zlib//:zlib",
    visibility = ["//visibility:public"],
)
""")

_llvm_zlib_compat = repository_rule(
    implementation = _llvm_zlib_compat_impl,
)

def _llvm_extension_impl(mctx):  # @unused
    _llvm_zlib_compat(name = "llvm_zlib")
    net_zstd(name = "llvm_zstd")
    llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "AMDGPU",
            "ARM",
            "NVPTX",
            "PowerPC",
            "RISCV",
            "SystemZ",
            "X86",
            "SPIRV",
        ],
    )

llvm_extension = module_extension(
    implementation = _llvm_extension_impl,
)
