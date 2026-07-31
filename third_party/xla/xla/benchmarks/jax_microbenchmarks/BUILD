"""JAX microbenchmarks for TPU performance tracking."""

load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

licenses(["notice"])

pytype_strict_library(
    name = "jax_profiler_utils",
    testonly = True,
    srcs = ["jax_profiler_utils.py"],
    deps = [
        "//third_party/py/pandas",
        "@absl_py//absl/logging",
        "@jax//jax",
    ],
)

pytype_strict_library(
    name = "matmul_lib",
    testonly = True,
    srcs = ["matmul_lib.py"],
    deps = [
        ":jax_profiler_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
        "@jax//jax",
    ],
)

py_test(
    name = "matmul_test",
    srcs = ["matmul_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":matmul_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@jax//jax",
    ],
)

py_test(
    name = "jax_profiler_utils_test",
    srcs = ["jax_profiler_utils_test.py"],
    deps = [
        ":jax_profiler_utils",
        "@absl_py//absl/testing:absltest",
        "@jax//jax",
    ],
)
