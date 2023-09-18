#include "graph.hpp"

namespace sky_infer {
    Graph::Graph(const std::string &param_path, const std::string &bin_path) {
        CHECK(!param_path_.empty() && !bin_path_.empty()) << "failed to initialise graph; bin path or parameter path cannot be empty";
        pnnx::Graph pnnx_graph;
        CHECK(pnnx_graph.load(param_path_, bin_path_) == 0) << "failed to initialise graph; invalid path: " << "bin " << bin_path_ << " param " << param_path_;
//        std::vector<pnnx::Operator*> pnnx_opts = pnnx_graph.ops;
//        CHECK(!pnnx_opts.empty()) << "failed to initialise graph; no operator exists in the model";
//      //  operators_.clear();


        GenerateLayers();

        for(auto* pnnx_opd: pnnx_graph.operands)
            operands_.insert({pnnx_opd->name, Operand(pnnx_opd)});

        for(auto* pnnx_opt: pnnx_graph.ops) {
            Operator opt = Operator(pnnx_opt);
            for(auto* pnnx_opd: pnnx_opt->inputs) {
                auto item = operands_.find(pnnx_opd->name);
                CHECK(item!=operands_.end()) << "failed to add input operand: " << pnnx_opd->name << " for operator: " << pnnx_opt->name << "; target operand object does not exist";
                opt.inputs_.push_back(&item->second);
            }
            for(auto* pnnx_opd: pnnx_opt->outputs) {
                auto item = operands_.find(pnnx_opd->name);
                CHECK(item!=operands_.end()) << "failed to add output operand: " << pnnx_opd->name << " for operator: " << pnnx_opt->name << "; target operand object does not exist";
                opt.outputs_.push_back(&item->second);
            }
//
            if(pnnx_opt->type == "pnnx.Input" || pnnx_opt->type == "pnnx.Output")
                opt.layer_ = layers_.at(LayerType::Dummy).get();
            else if(pnnx_opt->type == "nn.ReLU")
                opt.layer_ = layers_.at(LayerType::ReLU).get();
            else if(pnnx_opt->type == "pnnx.Expression")
                opt.layer_ = layers_.at(LayerType::Expression).get();
            else
              // TODO;
              ;

            operators_.insert({pnnx_opt->name, opt});
           // operands_.insert(pnnx_opt->name)
        }


        TopoSortOpts();

      //  std::set<std::string> runtime_opd_name_record;
    }

    void Graph::GenerateLayers() {
        for (int i = LayerType::Input; i <= LayerType::SoftMax; ++i) {
            auto type = static_cast<LayerType>(i);

            switch (type) {
                case LayerType::Input: {
                    if(layers_.find(LayerType::Input) == layers_.end())
                        layers_.insert({LayerType::Input, std::make_unique<LayerInput>()});
                    break;
                }

                case LayerType::Output: {
                    if(layers_.find(LayerType::Output) == layers_.end())
                        layers_.insert({LayerType::Output, std::make_unique<LayerOutput>()});
                    break;
                }

                case LayerType::ReLU: {
                    if(layers_.find(LayerType::ReLU) == layers_.end())
                        layers_.insert({LayerType::ReLU, std::make_unique<LayerReLU>()});
                    break;
                }

                case LayerType::Expression: {
                    if(layers_.find(LayerType::Expression) == layers_.end())
                        layers_.insert({LayerType::Expression, std::make_unique<LayerExpression>()});
                    break;
                }

                case LayerType::Linear: {
                    // TODO
                    break;
                }
                case LayerType::SoftMax: {
                    // TODO
                    break;
                }
                default:
                    LOG(ERROR) << "failed to create layer; unsupported layer type: " << type;
            }
       // layers_.insert({LayerType::ReLU, new LayerReLU()});
        }
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



//    Graph::~Graph() {
//        for(auto& item: layers_)
//            delete item.second;
//    }

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