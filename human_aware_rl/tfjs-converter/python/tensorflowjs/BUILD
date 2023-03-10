package(
    default_visibility = [
        "//tensorflowjs:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "tensorflowjs",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantization",
        ":version",
        "//tensorflowjs/converters:converter"
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_h5py_installed",
    # This is a dummy rule used as a h5py dependency in open-source.
    # We expect h5py to already be installed on the system, e.g. via
    # `pip install h5py`.
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`.
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a tensorflow dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow-gpu`.
)

py_library(
    name = "expect_tensorflow_hub_installed",
    # This is a dummy rule used as a tensorflow_hub dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow-gpu`.
)

py_library(
    name = "quantization",
    srcs = ["quantization.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_library(
    name = "write_weights",
    srcs = ["write_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantization",
        ":read_weights",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_library(
    name = "read_weights",
    srcs = ["read_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantization",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantization",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_test(
    name = "write_weights_test",
    srcs = ["write_weights_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":write_weights",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_test(
    name = "read_weights_test",
    srcs = ["read_weights_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":read_weights",
        ":write_weights",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

# A filegroup BUILD target that includes all the op list json files in the
# the op_list/ folder. The op_list folder itself is a symbolic link to the
# actual op_list folder under src/.
filegroup(
    name = "op_list_jsons",
    srcs = glob(["op_list/*.json"]),
)
