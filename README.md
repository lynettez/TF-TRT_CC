# Convert Graph Using TF-TRT C++ API 
Present a example using TF-TRT C++ API to do trt graph converting.

Verified with:
1. TensorFlow 1.14
2. Jetson AGX, JetPack 4.2.2
3. TensorRT 5.1.6
4. Bazel 0.25.3
5. JDK 11

This example can be built by Bazel with TensorFlow Source.\
Please follow the https://www.tensorflow.org/install/source to install Bazel and clone the TF Source.\
Make sure you checkout the branch to **r1.14**

Run configure before the fisrt time of using bazel build. \In this case, you should choose 'y' for the options related 'CUDA' and "TensorRT" during the configuration.
```
cd tensorflow
./configure
```

Copy 'example' folder to /tensorflow/cc/. \
Build the CC example, the first build would take about several hours.
```
bazel build --config=opt  --config=cuda //tensorflow/cc/example:example --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures
```

Trigger the converter to optimize the graph.
```
./bazel-bin/tensorflow/cc/example/example tensorflow/cc/example/mnist_frozen_graph.pb ArgMax,Softmax
```

If you want to see more INFO TF logs, you may
```
export TF_CPP_MIN_VLOG_LEVEL=2
```

