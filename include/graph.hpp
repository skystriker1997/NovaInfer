#ifndef SKY_INFER_GRAPH
#define SKY_INFER_GRAPH


#include "layer/concrete/convolution.hpp"
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



namespace sky_infer {

    class Graph {
    private:

        std::map<LayerType, std::function<std::shared_ptr<Layer>(pnnx::Operator*)>> layer_creators_;
        std::string param_path_;
        std::string bin_path_;
        std::map<std::string, std::shared_ptr<Layer>> layers_;
        std::map<std::string, std::shared_ptr<Batchf>> batches_;

        std::vector< std::shared_ptr<Layer>> topo_sorted_layers_;

        std::set<std::string> raw_inputs_;

        std::shared_ptr<Layer> CreateLayer(pnnx::Operator* opt);

        void TopoSortLayers();

        Check check_;


    public:
        Graph(const std::string& param_path, const std::string& bin_path);

        ~Graph() = default;

        void Ini();

        void Forward();

    };


    class LayerFactory {
    private:

       // Check check_;
        Eigen::MatrixXf AttrFormatConvert(std::vector<char>& raw);

    public:
        LayerFactory();
        std::shared_ptr<Layer> Create(pnnx::Operator*);
        ~LayerFactory() = default;
    };

}


#endif