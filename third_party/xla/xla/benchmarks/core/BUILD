"""Core microbenchmark framework for JAX on TPU."""

load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "platform_info",
    srcs = ["platform_info.py"],
    deps = [
        "//third_party/py/immutabledict",
        "@jax//jax",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "benchmark",
    testonly = True,
    srcs = ["benchmark.py"],
    deps = [
        "//third_party/py/numpy",
        "//xla/benchmarks/jax_microbenchmarks:jax_profiler_utils",
        "@absl_py//absl/logging",
        "@jax//jax",
    ],
)

py_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.py"],
    deps = [
        ":benchmark",
        "//third_party/py/numpy",
        "//xla/benchmarks/jax_microbenchmarks:jax_profiler_utils",
        "@absl_py//absl/testing:absltest",
        "@jax//jax",
    ],
)
