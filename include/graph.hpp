#ifndef NOVA_INFER_GRAPH
#define NOVA_INFER_GRAPH


#include "layer/concrete/convolution.hpp"
#include "layer/concrete/transposed_convolution.hpp"
#include "layer/concrete/expression.hpp"
#include "layer/concrete/maxpooling.hpp"
#include "layer/concrete/relu.hpp"
#include "layer/concrete/linear.hpp"
#include "layer/concrete/flatten.hpp"
#include "layer/concrete/sigmoid.hpp"


#include <queue>
#include <set>
#include <map>
#include <functional>
#include "pnnx/ir.h"



namespace nova_infer {

    class Graph {
    private:

        std::string param_path_;
        std::string bin_path_;

        std::map<std::string, std::shared_ptr<Layer>> layers_;

        std::map<std::string, std::vector<int>> data_node_shape_;
        std::map<std::string, std::shared_ptr<Batchf>> data_nodes_;

        std::vector< std::shared_ptr<Layer>> topo_sorted_layers_;

        std::string initial_data_;
        std::string final_output_;

        std::shared_ptr<Layer> CreateLayer(pnnx::Operator *opt);

        void TopoSortLayers();

        Check check_;

        int batch_size = 0;


    public:
        Graph(const std::string &param_path, const std::string &bin_path);

        ~Graph() = default;

        void AppendInput(const Tensor<float> &input);

        void Forward();

    };

}


#endif