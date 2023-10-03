#ifndef SKY_INFER_LAYER_MAXPOOLING
#define SKY_INFER_LAYER_MAXPOOLING


#include "layer/layer.hpp"

namespace sky_infer {

    class LayerMaxpooling: public Layer {

    private:
        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        std::vector<int> stride_;
        std::vector<int> padding_;
        std::vector<int> kernel_size_;


        friend class Graph;

    public:
        LayerMaxpooling(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output, std::vector<int> stride, std::vector<int> padding, std::vector<int> kernel_size);


        void Forward() override;


        ~LayerMaxpooling() override = default;

    };
}





#endif