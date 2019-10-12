#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

const int32 input_width = 299;
const int32 input_height = 299;
const float input_mean = 0;
const float input_std = 255;

const string input_layer = "image_tensor:0";
const vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };

inline std::vector<std::string> splitToStringVec(const std::string& option, char separator)
{
    std::vector<std::string> options;

    for(size_t start = 0; start < option.length(); )
    {
        size_t separatorIndex = option.find(separator, start);
        if (separatorIndex == std::string::npos)
        {
            separatorIndex = option.length();
        }
        options.emplace_back(option.substr(start, separatorIndex - start));
        start = separatorIndex + 1;
    }

    return options;
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = string(data);
  return Status::OK();
}


Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else {
    ;
  }
  auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

  auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
  return Status::OK();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Please specify the model file path, output names and input image path, like ./example PB_FILE_PATH OUTPUT1,OUTPUT2 PATH_TO_PGN" << "\n";
    return 1;
  }
  std::string modelPath = argv[1];
  std::string outputNames = argv[2];
  std::string image_path  = argv[3];

  // Read frozen graph from file
  GraphDef graph_def;
  Status read_status = ReadBinaryProto(Env::Default(), modelPath, &graph_def);
  if (!read_status.ok()) {
    std::cout << read_status.ToString() << "\n";
    return 1;
  }
  
  std::cout << "Origin graph has " << graph_def.node().size() << " nodes in total\n";

  std::vector<string> output_names{splitToStringVec(outputNames, ',')};
  GraphDef output_graph_def;

  // Specify the rewriter config to run TensorRTOptimizer
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("TensorRTOptimizer");
  // Specify the config of TRTOptimizationPass
  auto& parameters = *custom_config->mutable_parameter_map();
  parameters["minimum_segment_size"].set_i(1);
  parameters["max_batch_size"].set_i(1);
  parameters["is_dynamic_op"].set_b(true);
  parameters["max_workspace_size_bytes"].set_i(256LL << 20);
  parameters["precision_mode"].set_s("FP32");
  parameters["use_calibration"].set_b(false);

  grappler::MetaOptimizer optimizer(nullptr, config_proto);
  grappler::GrapplerItem item;
  item.fetch = output_names;
  item.graph.Swap(&graph_def);
  item.id = "tf_graph";
  grappler::GraphProperties graph_properties(item);

  const Status status_1 = optimizer.Optimize(nullptr, item, &output_graph_def);
  if (!status_1.ok()) {
    std::cout << status_1.ToString() << "\n";
    return 1;
  }

  // Print the optimizing result
  int num_trt_ops = 0;
  for (const NodeDef& node : output_graph_def.node()) {
    //std::cout << node.name() << "\n";
    if (node.name().find("TRTEngineOp") != std::string::npos) {
      ++num_trt_ops;
    }
  }
  std::cout << "After converting, nodes in total: " << output_graph_def.node().size() << "\n";
  std::cout << "Num of TRT nodes: " << num_trt_ops << "\n";

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];
  LOG(INFO) <<"image shape:" << resized_tensor.shape().DebugString()<< ",len:" << resized_tensors.size() << ",tensor type:"<< resized_tensor.dtype();

  // Actually run the image through the model.
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  status = session->Create(output_graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   output_layer, {}, &outputs);

  if (!run_status.ok()) {
      std::cout << "ERROR: RUN failed..."  << std::endl;
      std::cout << run_status.ToString() << "\n";
      return -1;
  }

  tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
  tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
  tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
  auto boxes = outputs[0].flat_outer_dims<float,3>();

  LOG(INFO) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();

  for(size_t i = 0; i < num_detections(0) && i < 20;++i)
  {
    if(scores(i) > 0.5)
    {
      LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
    }
  }

  return 0;
}
