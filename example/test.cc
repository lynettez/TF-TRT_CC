#include "tensorflow/c/c_api.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <string>
#include <cstdlib>
#include <iostream>
using namespace tensorflow;
using tensorflow::Status;

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

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Please specify the model file path and output names, like ./example PB_FILE_PATH OUTPUT1,OUTPUT2" << "\n";
    return 1;
  }
  std::string modelPath = argv[1];
  std::string outputNames = argv[2];

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read frozen graph from file
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), modelPath, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
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

  // Print see the optimizing result
  int num_trt_ops = 0;
  for (const NodeDef& node : output_graph_def.node()) {
    std::cout << node.name() << "\n";
    if (node.name().find("TRTEngineOp") != std::string::npos) {
      ++num_trt_ops;
    }
  }
  std::cout << "After converting, nodes in total: " << output_graph_def.node().size() << "\n";
  std::cout << "Num of TRT nodes: " << num_trt_ops << "\n";

  return 0;
}
