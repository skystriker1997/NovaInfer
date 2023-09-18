#ifndef SKY_INFER_GRAPH
#define SKY_INFER_GRAPH

#include <memory>
#include "operator.hpp"
#include <glog/logging.h>
#include "tensor/tensor.hpp"

#include "layer/layer.hpp"
#include "layer/concrete/relu.hpp"
#include "layer/concrete/expression/expression.hpp"
#include "layer/concrete/input.hpp"
#include "layer/concrete/output.hpp"


#include <queue>



namespace sky_infer {

        class Graph {
        private:
            std::string param_path_;
            std::string bin_path_;
            std::map<LayerType, std::unique_ptr<Layer>> layers_;
            std::map<std::string, Operand> operands_;
            std::map<std::string, Operator> operators_;

            std::vector<Operator*> topo_sorted_operators_;

        public:
            Graph(const std::string& param_path, const std::string& bin_path);

            ~Graph() = default;

            void IniInputData(const std::string& opt_name, std::vector<Tensor<float>>& data);

            void Forward();

            void GenerateLayers();

            void TopoSortOpts();

        };



}


#endif