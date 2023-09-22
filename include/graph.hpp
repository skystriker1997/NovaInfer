#ifndef SKY_INFER_GRAPH
#define SKY_INFER_GRAPH


#include "layer/layer_factory.hpp"
#include <queue>



namespace sky_infer {

        class Graph {
        private:
            std::string param_path_;
            std::string bin_path_;
            LayerFactory layer_factory_;
            std::map<std::string, Operand> operands_;
            std::map<std::string, Operator> operators_;

            std::vector<Operator*> topo_sorted_operators_;

        public:
            Graph(const std::string& param_path, const std::string& bin_path);

            ~Graph() = default;

            void IniInputData(const std::string& opt_name, std::vector<Tensor<float>>& data);

            void Forward();

            void TopoSortOpts();

        };



}


#endif