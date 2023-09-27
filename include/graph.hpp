#ifndef SKY_INFER_GRAPH
#define SKY_INFER_GRAPH


#include "layer/layer.hpp"
#include "layer/concrete/convolution.hpp"
#include "layer/concrete/expression.hpp"
#include "layer/concrete/maxpooling.hpp"
#include "layer/concrete/relu.hpp"


#include <queue>
#include <set>
#include <map>
#include "pnnx/ir.h"



namespace sky_infer {

    class Graph {
    private:

        std::string param_path_;
        std::string bin_path_;
        std::map<std::string, std::shared_ptr<Layer>> layers_;
        std::map<std::string, std::shared_ptr<Batch<float>>> data_nodes_;

        std::vector< std::shared_ptr<Layer>> topo_sorted_layers_;

        std::set<std::string> raw_inputs_;

        std::shared_ptr<Layer> CreateLayer(pnnx::Operator* opt);

        void TopoSortLayers();

        Check check_;


    public:
        Graph(const std::string& param_path, const std::string& bin_path);

        ~Graph() = default;

        void Forward();

    };
}


#endif