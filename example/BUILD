load("//tensorflow:tensorflow.bzl", "tf_cc_binary")


tf_cc_binary(
    name = "example",
    srcs = ["test.cc"],
    deps = [
	"//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
	"//tensorflow/compiler/tf2tensorrt:trt_conversion",
        "//tensorflow/compiler/tf2tensorrt:trt_engine_op_op_lib",
	"//tensorflow/compiler/tf2tensorrt:trt_op_kernels",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core:protos_all_cc"
    ],
)
