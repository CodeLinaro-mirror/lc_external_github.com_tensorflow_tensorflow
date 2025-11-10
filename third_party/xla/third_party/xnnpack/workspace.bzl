"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "4403c0d4ab7d4bb2e38e60fc15907b69b0e023a86ad6f72d94a1214bd62b4754",
        strip_prefix = "XNNPACK-63ab808582d53a6626cf498085d7fa5a61b622ee",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/63ab808582d53a6626cf498085d7fa5a61b622ee.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
