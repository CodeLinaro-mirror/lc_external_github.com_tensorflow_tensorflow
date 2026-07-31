load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "cost_model",
    srcs = ["cost_model.py"],
    deps = [
        "//third_party/py/numpy",
        "//xla/benchmarks/core:platform_info",
        "@jax//jax",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "memory_utils",
    srcs = ["memory_utils.py"],
    deps = [
        "@jax//jax",
        "@jax//jax/experimental:pallas",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "dense_matmul_lib",
    testonly = True,
    srcs = ["dense_matmul_lib.py"],
    deps = [
        ":memory_utils",
        "//xla/benchmarks/core:benchmark",
        "@absl_py//absl/logging",
        "@jax//jax",
        "@jax//jax/experimental:pallas",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

py_test(
    name = "dense_matmul",
    timeout = "long",
    srcs = ["dense_matmul.py"],
    tags = [
        # Tests require arguments to be provided.
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_model",
        ":dense_matmul_lib",
        "//xla/benchmarks/core:platform_info",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@jax//jax",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "subchannel_matmul_lib",
    testonly = True,
    srcs = ["subchannel_matmul_lib.py"],
    deps = [
        ":memory_utils",
        "//xla/benchmarks/core:benchmark",
        "@jax//jax",
        "@jax//jax/experimental:layout",
        "@jax//jax/experimental:pallas",
        "@jax//jax/experimental:pallas_tpu",
    ],
)

py_test(
    name = "subchannel_matmul",
    timeout = "long",
    srcs = ["subchannel_matmul.py"],
    tags = [
        # Tests require arguments to be provided.
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_model",
        ":subchannel_matmul_lib",
        "//third_party/py/immutabledict",
        "//xla/benchmarks/core:platform_info",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@jax//jax",
        "@jax//jax/experimental:pallas_tpu",
    ],
)
