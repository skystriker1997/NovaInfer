#include "graph.hpp"


namespace nova_infer {

    Graph::Graph(std::string_view param_path, std::string_view bin_path) {

        param_path_ = param_path;
        bin_path_ = bin_path;

        check_(!param_path_.empty() && !bin_path_.empty())
                << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;

        check_(pnnx_graph.load(param_path_, bin_path_) == 0)
                << "failed to initialise graph; failed to load the model and coefficients" + std::string ("\n") + "bin: " + bin_path_ + std::string ("\n") + "param: " + param_path_;

        auto check_shape = [](const std::vector<int> &raw_shape, Check &check) -> std::vector<int> {
            for(int i: raw_shape)
                check(i>0) << "failed to initialise graph; each dimension of tensor should be a positive integer";
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
                    check(raw_shape[i] == 1) << "failed to initialise graph; tensor should be 3-dimensional";
                }
                shape = {raw_shape[n-3], raw_shape[n-2], raw_shape[n-1]};
            }
            return shape;
        };

        for(pnnx::Operand *opd: pnnx_graph.operands) {
            check_(opd->type == 1) << "failed to initialise graph; failed to construct tensor; only support float32";
            std::vector<int> shape = check_shape(opd->shape, check_);
            tensor_shape_.insert({opd->name, shape});
            tensors_.insert({opd->name, std::make_shared<Batchf>()});
            if(opd->producer->type == "pnnx.Input") {
                check_(input_tensor_.empty()) << "failed to initialise graph; failed to construct tensor; allow only one input";
                input_tensor_ = opd->name;
            }
            if(opd->consumers[0]->type == "pnnx.Output") {
                check_(output_tensor_.empty()) << "failed to initialise graph; failed to construct tensor; allow only one input";
                output_tensor_ = opd->name;
            }
        }

        for(pnnx::Operator *opt: pnnx_graph.ops) {
            if(opt->type == "pnnx.Input" || opt->type == "pnnx.Output")
                continue;
            check_(layers_.find(opt->name) == layers_.end()) << "failed to initialise graph; layers' name should not duplicate";
            layers_.insert({opt->name, CreateLayer(opt)});
        }

        TopoSortLayers();

    }



    std::shared_ptr<Layer> Graph::CreateLayer(pnnx::Operator *opt) {

        if(opt->type == "nn.ReLU") {

            return MakeLayerReLU(opt);

        } else if(opt->type == "Expression") {

            return MakeLayerExpression(opt);

        } else if(opt->type == "nn.MaxPool2d") {

            return MakeLayerMaxpooling(opt);

        } else if(opt->type == "torch.flatten") {

            return MakeLayerFlatten(opt);

        } else if(opt->type == "nn.Linear") {

            return MakeLayerLinear(opt);

        } else if(opt->type == "nn.Sigmoid") {

            return MakeLayerSigmoid(opt);

        } else if(opt->type == "nn.Conv2d") {

            return MakeLayerConvolution(opt);

        } else if(opt->type == "nn.ConvTranspose2d") {

            return MakeLayerTransposedConvolution(opt);

        } else if(opt->type == "torch.cat") {

            return MakeLayerCat(opt);

        } else if(opt->type == "nn.Softmax") {

            return MakeLayerSoftmax(opt);

        } else {

            check_(false) << "failed to create layer; unsupported layer type: " + opt->type;
            return nullptr;

        }
    }


    void Graph::TopoSortLayers() {

        std::queue<std::shared_ptr<Layer>> topo_queue;

        std::map<std::string, int> in_degrees;

        std::map<std::string, std::vector<std::shared_ptr<Layer>>> consumers;

        for(auto &[name, layer]: layers_) {
            for(const auto &input: layer->GetInputName()) {
                consumers[input].emplace_back(layer);
            }
        }

        for(auto &[name, layer]: layers_) {
            bool first = true;
            int in_degree = 0;
            for(const auto &input: layer->GetInputName()) {
                if(input != input_tensor_) {
                    first = false;
                    in_degree++;
                }
            }
            if(first)
                topo_queue.push(layer);
            in_degrees.insert({layer->GetName(), in_degree});
        }

        while(!topo_queue.empty()) {
            auto &layer = topo_queue.front();
            for(const auto &output: layer->GetOutputName()) {
                for(auto &consumer: consumers[output]) {
                    int &in_degree = in_degrees[consumer->GetName()];
                    in_degree--;
                    if(!in_degree) {
                        topo_queue.push(consumer);
                    }
                }
            }
            topo_sorted_layers_.emplace_back(layer);
            topo_queue.pop();
        }

        check_(topo_sorted_layers_.size() == layers_.size())
                << "failed to topo sort layers; the pnnx graph is not a directed acyclic one";

    }


    void Graph::AppendBatch(const Tensor<float> &input) {
        for(auto &[name, batch]: tensors_) {
            if(name == input_tensor_) {
                std::vector<int> shape = {input.Channels(), input.Rows(), input.Cols()};
                check_(tensor_shape_.at(input_tensor_) == shape) << "failed to set up input; the shape of input tensor is not as expected";
                batch->emplace_back(input);
            } else {
                batch->emplace_back((tensor_shape_.at(name)));
            }
        }
        batch_size++;
    }


    std::shared_ptr<Batchf> Graph::GetOutput() {
        return tensors_.at(output_tensor_);
    }


    void Graph::Forward() {
        check_(batch_size > 0) << "failed to forward the graph; need at least one element in batch";
        for(const auto &layer: topo_sorted_layers_) {
            if(layer->GetType()==LayerType::Expression || layer->GetType()==LayerType::Cat) {
                for(const auto &name: layer->GetInputName()) {
                    auto item = tensors_.find(name);
                    check_(item != tensors_.end())
                            << "failed to forward the graph; " + layer->GetName() + " failed to find tensor " + name + " to bind with";
                    layer->AttachInput(item->second);
                }
            } else {
                auto item = tensors_.find(layer->GetInputName()[0]);
                check_(item != tensors_.end())
                        << "failed to forward the graph; " + layer->GetName() + " failed to find tensor " + layer->GetInputName()[0] + " to bind with";
                layer->AttachInput(item->second);
            }
            auto item = tensors_.find(layer->GetOutputName()[0]);
            check_(item != tensors_.end())
                    << "failed to forward the graph; " + layer->GetName() + " failed to find tensor " + layer->GetInputName()[0] + " to bind with";
            layer->AttachOutput(item->second);
        }

        for(const auto &layer: topo_sorted_layers_) {
            std::cout << "layer " << layer->GetName() << " started" << std::endl;
            layer->Forward();
            std::cout << "layer " << layer->GetName() << " completed" << std::endl;
        }

    }




}