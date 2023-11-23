#include "graph.hpp"


namespace nova_infer {

    Graph::Graph(const std::string &param_path, const std::string &bin_path): param_path_(param_path), bin_path_(bin_path) {

        check_(!param_path_.empty() && !bin_path_.empty())
                << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;

        check_(pnnx_graph.load(param_path_, bin_path_) == 0)
                << "failed to initialise graph; invalid path: " + std::string("bin ") + bin_path_ + " param " + param_path_;

        auto check_shape = [](const std::vector<int> &raw_shape, Check &check) -> std::vector<int> {
            for(int i: raw_shape)
                check(i>0) << "failed to initialise graph; elements of operand's shape vector should be positive int";
            std::vector<int> shape(3);
            auto n = raw_shape.size();
            if(n == 1) {
                shape = {1, 1, raw_shape[0]};
            } else if(n == 2) {
                shape = {1, raw_shape[0], raw_shape[1]};
            } else if(n == 3) {
                shape = raw_shape;
            } else {
                for(int i=0; i<n-3; i++) {
                    check(raw_shape[i] == 1) << "failed to initialise graph; operand should be 3-dimensional";
                }
                shape = {raw_shape[n-3], raw_shape[n-2], raw_shape[n-1]};
            }
            return shape;
        };

        for (pnnx::Operand *opd: pnnx_graph.operands) {
            check_(opd->type == 1) << "failed to construct data node; only support float32";
            std::vector<int> shape = check_shape(opd->shape, check_);
            data_node_shape_.insert({opd->name, shape});
            data_nodes_.insert({opd->name, std::make_shared<Batchf>()});
            if (opd->producer->type == "pnnx_Input") {
                check_(initial_data_.empty()) << "failed to construct data node; allow only one initial data";
                initial_data_ = opd->name;
            }
            if (opd->consumers[0]->type == "pnnx_Output") {
                check_(final_output_.empty()) << "failed to construct data node; allow only one final output";
                final_output_ = opd->name;
            }
        }

        for (pnnx::Operator *opt: pnnx_graph.ops) {
            if (opt->type == "pnnx_Input" || opt->type == "pnnx_Output")
                continue;
            check_(layers_.find(opt->name) == layers_.end()) << "failed to add layer; duplicate layer name";
            layers_.insert({opt->name, CreateLayer(opt)});
        }

        TopoSortLayers();

    }



    std::shared_ptr<Layer> Graph::CreateLayer(pnnx::Operator *opt) {

        if (opt->type == "nn.ReLU") {

            return MakeLayerReLU(opt);

        } else if (opt->type == "Expression") {

            return MakeLayerExpression(opt);

        } else if (opt->type == "nn.MaxPool2d") {

            return MakeLayerMaxpooling(opt);

        } else if (opt->type == "torch.flatten") {

            return MakeLayerFlatten(opt);

        } else if (opt->type == "nn.Linear") {

            return MakeLayerLinear(opt);

        } else if (opt->type == "nn.Sigmoid") {

            return MakeLayerSigmoid(opt);

        } else if (opt->type == "nn.Conv2d") {

            return MakeLayerConvolution(opt);

        } else if (opt->type == "nn.ConvTranspose2d") {

            return MakeLayerTransposedConvolution(opt);

        } else {

            check_(false) << "failed to create layer; unsupported layer type: " + opt->type;

        }
    }


    void Graph::TopoSortLayers() {

        std::queue<std::shared_ptr<Layer>> topo_queue;

        std::map<std::string, int> in_degrees;

        std::map<std::string, std::vector<std::shared_ptr<Layer>>> consumers;

        for (auto &[name, layer]: layers_) {
            for (const auto &input: layer->GetInputName())
                consumers[input].emplace_back(layer);
        }

        for (auto &[name, layer]: layers_) {
            bool first = true;
            int in_degree = 0;
            for (const auto &input: layer->GetInputName()) {
                if (input == initial_data_) {
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


    void Graph::AppendInput(const Tensor<float> &input) {
        for (auto &[name, batch]: data_nodes_) {
            if (name == initial_data_) {
                std::vector<int> shape = {input.Channels(), input.Rows(), input.Cols()};
                check_(data_node_shape_.at(initial_data_) == shape) << "failed to push input; the shape of input tensor is not as expected";
                batch->emplace_back(input);
            } else {
                batch->emplace_back((data_node_shape_.at(name)));
            }
        }
        batch_size++;
    }


    void Graph::Forward() {
        if (batch_size > 0) {
            for(const auto &layer: topo_sorted_layers_) {
                if(layer->GetType() == LayerType::Expression) {
                    std::vector<std::shared_ptr<Batchf>> inputs;
                    std::shared_ptr<Batchf> output;
                    for(const auto &name: layer->GetInputName()) {
                        inputs.emplace_back(data_nodes_.at(name));
                    }
                    std::dynamic_pointer_cast<LayerExpression>(layer)->AssignInputs(inputs);
                    layer->AssignOutput(data_nodes_.at(layer->GetOutputName()[0]));
                } else {
                    layer->AssignInput(data_nodes_.at(layer->GetInputName()[0]));
                    layer->AssignOutput(data_nodes_.at(layer->GetOutputName()[0]));
                }
            }

            for (const auto &layer: topo_sorted_layers_)
                layer->Forward();
        }
    }



}