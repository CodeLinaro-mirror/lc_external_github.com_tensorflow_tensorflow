"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "15d46f2efd8c331261452861a58d604fb26dae0a6406ef7bc48d3f6ef579400d",
        strip_prefix = "XNNPACK-1187a01646e94414b0753e6eb749fb0b238bd365",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/1187a01646e94414b0753e6eb749fb0b238bd365.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
