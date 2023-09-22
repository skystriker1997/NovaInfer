#include "graph.hpp"

namespace sky_infer {
    Graph::Graph(const std::string &param_path, const std::string &bin_path) {
        CHECK(!param_path_.empty() && !bin_path_.empty()) << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;
        CHECK(pnnx_graph.load(param_path_, bin_path_) == 0) << "failed to initialise graph; invalid path: " << "bin " << bin_path_ << " param " << param_path_;


        for(auto* pnnx_opd: pnnx_graph.operands)
            operands_.insert({pnnx_opd->name, Operand(pnnx_opd)});



        for(auto* pnnx_opt: pnnx_graph.ops) {
            Operator opt = Operator(pnnx_opt);
            for(auto* pnnx_opd: pnnx_opt->inputs) {
                auto item = operands_.find(pnnx_opd->name);
                CHECK(item!=operands_.end()) << "failed to add input operand: " << pnnx_opd->name << " for operator: " << pnnx_opt->name << "; target operand object does not exist";
                opt.inputs_.emplace_back(&item->second);
            }
            for(auto* pnnx_opd: pnnx_opt->outputs) {
                auto item = operands_.find(pnnx_opd->name);
                CHECK(item!=operands_.end()) << "failed to add output operand: " << pnnx_opd->name << " for operator: " << pnnx_opt->name << "; target operand object does not exist";
                opt.outputs_.emplace_back(&item->second);
            }
//
            if(pnnx_opt->type == "pnnx.Input")
                opt.layer_ = layer_factory_.GetLayer(LayerType::Input);
            else if(pnnx_opt->type == "pnnx.Output")
                opt.layer_ = layer_factory_.GetLayer(LayerType::Output);
            else if(pnnx_opt->type == "nn.ReLU")
                opt.layer_ = layer_factory_.GetLayer(LayerType::ReLU);
            else if(pnnx_opt->type == "pnnx.Expression")
                opt.layer_ = layer_factory_.GetLayer(LayerType::Expression);
            else
              // TODO;
              ;

            operators_.insert({pnnx_opt->name, opt});
        }


        TopoSortOpts();

    }




    void Graph::IniInputData(const std::string &opt_name, std::vector<Tensor<float>> &data) {
        auto item = operators_.find(opt_name);
        CHECK(item!=operators_.end() && item->second.type_=="pnnx.Input") << "failed to initialise input data; wrong operator name";
        item->second.outputs_[0]->data_ = std::move(data);
    }


    void Graph::Forward() {
        for(auto* opt: topo_sorted_operators_)
            opt->layer_->Forward(opt);
    }



    void Graph::TopoSortOpts() {

        std::queue<Operator*> topo_queue;

        std::map<std::string, int> in_degree;

        std::map<std::string, std::vector<Operator*>> consumers;

        for(auto& item: operators_) {
            for(auto* opd: item.second.inputs_)
                consumers[opd->name_].push_back(&item.second);
        }

        for(auto& [name, opt] : operators_) {
            if(opt.type_ == "pnnx.Input") {
                topo_queue.push(&opt);
                in_degree[name] = 0;
            } else {
                in_degree[name] = opt.inputs_.size();
            }
        }

        while(!topo_queue.empty()) {
            Operator* opt = topo_queue.front();
            for(auto& output : opt->outputs_) {
                for(auto consumer: consumers[output->name_]) {
                    in_degree[consumer->name_] -= 1;
                    if(!in_degree[consumer->name_])
                        topo_queue.push(consumer);
                }
            }
            topo_sorted_operators_.push_back(opt);
            topo_queue.pop();
        }

        CHECK(topo_sorted_operators_.size() == operators_.size()) << "failed to topo sort operators; the original graph is not a directed acyclic one";

    }


}