#ifndef SKY_INFER_LAYER_FLATTEN
#define SKY_INFER_LAYER_FLATTEN


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerFlatten: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int start_dim_;
        int end_dim_;

    public:

        LayerFlatten(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output, int start_dim, int end_dim);

        void Forward() override;

        ~LayerFlatten() override = default;
    };
}




#endif