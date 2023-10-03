#include "graph.hpp"


namespace sky_infer {

    Graph::Graph(const std::string &param_path, const std::string &bin_path): param_path_(param_path), bin_path_(bin_path) {

        check_(!param_path_.empty() && !bin_path_.empty())
                << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;

        check_(pnnx_graph.load(param_path_, bin_path_) == 0)
                << "failed to initialise graph; invalid path: " + std::string("bin ") + bin_path_ + " param " +
                   param_path_;


        auto refine_shape = [](const std::vector<int>& raw_shape, Check& check) {
            check(!raw_shape.empty() && raw_shape.size()<=4) << "failed to initialise graph; size of operand's shape vector should be between 1 and 4 (inclusive)";
            for(int i: raw_shape)
                check(i>0) << "failed to initialise graph; elements of operand's shape vector should be positive int";
            std::vector<int> shape(4);
            auto n = raw_shape.size();
            if(n == 1)
                shape = {1, 1, 1, raw_shape[0]};
            else if(n == 2)
                shape = {1, 1, raw_shape[0], raw_shape[1]};
            else if(n == 3)
                shape = {1, raw_shape[0], raw_shape[1], raw_shape[2]};
            else
                shape = raw_shape;
            return shape;
        };


        for (auto *opd: pnnx_graph.operands) {
            check_(opd->type == 1) << "failed to construct data node; only support float32";
            std::vector<int> shape = refine_shape(opd->shape, check_);
            batches_.insert({opd->name, std::make_shared<Batchf>(shape[0], Tensor<float>{std::vector<int>{shape[1], shape[2], shape[3]}})});
            if (opd->producer->type == "pnnx_Input")
                raw_inputs_.insert(opd->name);
        }

        for (auto *opt: pnnx_graph.ops) {
            if (opt->type == "pnnx_Input" || opt->type == "pnnx_Output")
                continue;
            check_(layers_.find(opt->name) == layers_.end()) << "failed to add layer; duplicate layer name";
            layers_.insert({opt->name, CreateLayer(opt)});
        }

        TopoSortLayers();

    }





    std::shared_ptr<Layer> Graph::CreateLayer(pnnx::Operator *opt) {

        std::vector<std::shared_ptr<Batchf>> inputs;
        for(auto* opd: opt->inputs) {
            auto in = batches_.find(opd->name);
            check_(in != batches_.end()) << "failed to create layer; cannot find corresponding input";
            inputs.push_back(in->second);
        }

        std::vector<std::shared_ptr<Batchf>> outputs;
        for(auto* opd: opt->outputs) {
            auto out = batches_.find(opd->name);
            check_(out != batches_.end()) << "failed to create layer; cannot find corresponding output";
            outputs.push_back(out->second);
        }

            // relu
        if (opt->type == "nn.ReLU") {
            check_(inputs.size()==1 && outputs.size()==1) << "failed to create layer relu; only support one batch as input and one batch as output";
            return std::make_shared<LayerReLU>(std::move(opt->name), inputs[0], outputs[0]);
        }

            // expression
        else if (opt->type == "Expression") {

            check_(!inputs.empty() && outputs.size()==1) << "failed to create layer expression; only support one batch as output; at least one batch as input";

            auto expr = opt->params.find("expr");

            check_(expr != opt->params.end()) << "failed to create expression layer; miss parameter expression";

            return std::make_shared<LayerExpression>(std::move(opt->name), std::move(inputs), outputs[0], std::move(expr->second.s));
        }

            // maxpooling
        else if (opt->type == "nn.MaxPool2d") {
            check_(inputs.size()==1 && outputs.size()==1) << "failed to create layer maxpooling; only support one batch as input and one batch as output";

            auto stride = opt->params.find("stride");
            check_(stride != opt->params.end()) << "failed to create maxpooling layer; miss parameter stride";
            auto padding = opt->params.find("padding");
            check_(padding != opt->params.end()) << "failed to create maxpooling layer; miss parameter padding";
            auto kernel_size = opt->params.find("kernel_size");
            check_(kernel_size != opt->params.end()) << "failed to create maxpooling layer; miss parameter kernel_size";


            return std::make_shared<LayerMaxpooling>(std::move(opt->name), inputs[0], outputs[0],
                                                     std::move(stride->second.ai),
                                                     std::move(padding->second.ai),
                                                     std::move(kernel_size->second.ai));

        }

        // flatten

        else if (opt->type == "torch.flatten") {
            check_(inputs.size()==1 && outputs.size()==1) << "failed to create layer flatten; only support one batch as input and one batch as output";

            auto start_dim = opt->params.find("start_dim");
            check_(start_dim != opt->params.end()) << "failed to create flatten layer; miss parameter start_dim";

            auto end_dim = opt->params.find("end_dim");
            check_(end_dim != opt->params.end()) << "failed to create flatten layer; miss parameter end_dim";

            return std::make_shared<LayerFlatten>(std::move(opt->name), inputs[0], outputs[0],
                                                  start_dim->second.i, end_dim->second.i);
        }


        else if (opt->type == "nn.Linear") {
            check_(inputs.size()==1 && outputs.size()==1) << "failed to create layer linear; only support one batch as input and one batch as output";

            auto convert_to_float = [](std::vector<char>& attr_val, Check& check){
                std::vector<float> vect_float;
                auto float_size = sizeof(float);
                check(attr_val.size() % float_size == 0) << "failed to convert char arr to float arr; total bytes cannot be divided by size of a float";
                for(auto i=0; i<attr_val.size()/float_size; i++) {
                    float f = *((float*)attr_val.data()+i);
                    vect_float.push_back(f);
                }
                return vect_float;
            };

            auto use_bias = opt->params.find("bias");
            check_(use_bias != opt->params.end()) << "failed to create linear layer; miss parameter bias";
            Eigen::VectorXf bias;

            if(use_bias->second.b) {
                auto item = opt->attrs.find("bias");
                check_(item != opt->attrs.end()) << "failed to create linear layer; miss attribute bias";

                std::vector<int>& shape = item->second.shape;

                check_(!shape.empty()) << "failed to create linear layer; bias should be 1-dimensional";
                if(shape.size()>1) {
                    for(int i=0; i<shape.size()-1; i++)
                        check_(shape[i]==1) << "failed to create linear layer; bias should be 1-dimensional";
                }

                std::vector<float> bias_f = convert_to_float(item->second.data, check_);

                bias = Eigen::Map<Eigen::RowVectorXf>(bias_f.data(), shape.back());

            }

            auto item = opt->attrs.find("weight");
            check_(item != opt->attrs.end()) << "failed to create linear layer; miss attribute weight";

            std::vector<int>& shape = item->second.shape;
            check_(!shape.empty()) << "failed to create linear layer; weights should be 2-dimensional";
            if(shape.size()>2) {
                for(int i=0; i<shape.size()-2; i++)
                    check_(shape[i]==1) << "failed to create linear layer; weights should be 2-dimensional";
            }

            std::vector<float> weights_f =  convert_to_float(item->second.data, check_);
            Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights = Eigen::Map<Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(weights_f.data(),shape[shape.size()-2],shape[shape.size()-1]);


            return std::make_shared<LayerLinear>(std::move(opt->name), inputs[0], outputs[0],
                                                       weights, use_bias->second.b, bias);

        }
        else if(opt->type == "nn.Sigmoid") {
            check_(inputs.size()==1 && outputs.size()==1) << "failed to create layer sigmoid; only support one batch as input and one batch as output";
            return std::make_shared<LayerSigmoid>(std::move(opt->name), inputs[0], outputs[0]);
        }
    }



    void Graph::Forward() {
        for (const auto &layer: topo_sorted_layers_)
            layer->Forward();
    }


    void Graph::TopoSortLayers() {

        std::queue<std::shared_ptr<Layer>> topo_queue;

        std::map<std::string, int> in_degrees;

        std::map<std::string, std::vector<std::shared_ptr<Layer>>> consumers;

        for (auto &[name, layer]: layers_) {
            for (const auto &input: layer->GetInputName())
                consumers[input].push_back(layer);
        }

        for (auto &[name, layer]: layers_) {
            bool first = true;
            int in_degree = 0;
            for (const auto &input: layer->GetInputName()) {
                if (raw_inputs_.find(input) == raw_inputs_.end()) {
                    first = false;
                    in_degree++;
                }
            }
            if (first)
                topo_queue.push(layer);
            in_degrees.insert({layer->GetName(), in_degree});
        }

        while (!topo_queue.empty()) {
            auto &layer = topo_queue.front();
            for(const auto& output: layer->GetOutputName()) {
                for (auto &consumer: consumers[output]) {
                    int &in_degree = in_degrees[consumer->GetName()];
                    in_degree--;
                    if (!in_degree)
                        topo_queue.push(consumer);
                }
            }
            topo_sorted_layers_.emplace_back(layer);
            topo_queue.pop();
        }

        check_(topo_sorted_layers_.size() == layers_.size())
                << "failed to topo sort layers; the pnnx graph is not a directed acyclic one";

    }


}