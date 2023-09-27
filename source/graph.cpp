#include "graph.hpp"


namespace sky_infer {

    Graph::Graph(const std::string &param_path, const std::string &bin_path) {
        check_(!param_path_.empty() && !bin_path_.empty())
                << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;
        check_(pnnx_graph.load(param_path_, bin_path_) == 0)
                << "failed to initialise graph; invalid path: " + std::string("bin ") + bin_path_ + " param " +
                   param_path_;


        for (auto *opd: pnnx_graph.operands) {
            check_(opd->type == 1) << "failed to construct data node; only support float32";
            data_nodes_.insert({opd->name, std::make_shared<Batch<float>>(opd->name, opd->shape)});
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

        // input && output
        if (opt->type == "pnnx_Input" || opt->type == "pnnx_Output")
            return {nullptr};

            // relu
        else if (opt->type == "nn.ReLU") {
            check_(opt->inputs.size() == 1 && opt->outputs.size() == 1)
                    << "failed to create relu layer; only support one input and one output";
            auto in = data_nodes_.find(opt->inputs[0]->name);
            check_(in != data_nodes_.end()) << "failed to create relu layer; cannot find corresponding input";
            auto out = data_nodes_.find(opt->outputs[0]->name);
            check_(out != data_nodes_.end()) << "failed to create relu layer; cannot find corresponding output";
            return std::make_shared<LayerReLU>(in->second, out->second);
        }

            // expression
        else if (opt->type == "Expression") {
            auto out = data_nodes_.find(opt->outputs[0]->name);
            check_(out != data_nodes_.end()) << "failed to create expression layer; cannot find corresponding output";
            auto expr = opt->params.find("expr");
            check_(expr != opt->params.end()) << "failed to create expression layer; miss parameter expression";
            std::vector<std::shared_ptr<Batch<float>>> ins;
            for (auto *opd: opt->inputs) {
                auto in = data_nodes_.find(opd->name);
                check_(in != data_nodes_.end()) << "failed to create expression layer; cannot find corresponding input";
                ins.push_back(in->second);
            }

            return std::make_shared<LayerExpression>(std::move(ins), out->second, std::move(expr->second.s));
        }

            // maxpooling
        else if (opt->type == "nn.MaxPool2d") {
            check_(opt->inputs.size() == 1 && opt->outputs.size() == 1)
                    << "failed to create maxpooling layer; only support one input and one output";
            auto stride = opt->params.find("stride");
            check_(stride != opt->params.end()) << "failed to create maxpooling layer; miss parameter stride";
            auto padding = opt->params.find("padding");
            check_(padding != opt->params.end()) << "failed to create maxpooling layer; miss parameter padding";
            auto kernel_size = opt->params.find("kernel_size");
            check_(kernel_size != opt->params.end()) << "failed to create maxpooling layer; miss parameter kernel_size";
            auto in = data_nodes_.find(opt->inputs[0]->name);
            check_(in != data_nodes_.end()) << "failed to create maxpooling layer; cannot find corresponding input";
            auto out = data_nodes_.find(opt->outputs[0]->name);
            check_(out != data_nodes_.end()) << "failed to create maxpooling layer; cannot find corresponding output";

            return std::make_shared<LayerMaxpooling>(in->second, out->second, std::move(stride->second.ai),
                                                     std::move(padding->second.ai), std::move(kernel_size->second.ai));

        }

        // TODO


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
            std::vector<std::shared_ptr<Batch<float>>> inputs = layer->GetInputs();
            for (auto &input: inputs)
                consumers[input->name_].push_back(layer);
        }

        for (auto &[name, layer]: layers_) {
            bool first = true;
            int in_degree = 0;
            std::vector<std::shared_ptr<Batch<float>>> inputs = layer->GetInputs();
            for (auto &input: inputs) {
                if (raw_inputs_.find(input->name_) == raw_inputs_.end()) {
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
            for (auto &consumer: consumers[layer->GetOutput()->name_]) {
                int &in_degree = in_degrees[consumer->GetName()];
                in_degree -= 1;
                if (!in_degree)
                    topo_queue.push(consumer);
            }
            topo_sorted_layers_.emplace_back(layer);
            topo_queue.pop();
        }

        check_(topo_sorted_layers_.size() == layers_.size())
                << "failed to topo sort layers; the pnnx graph is not a directed acyclic one";

    }

}