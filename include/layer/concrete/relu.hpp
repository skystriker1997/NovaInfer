#ifndef SKY_INFER_LAYER_RELU
#define SKY_INFER_LAYER_RELU


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerReLU: public Layer {
    private:

        LayerType type_;
        std::shared_ptr<Batch<float>> input_;
        std::shared_ptr<Batch<float>> output_;
        Check check_;

    public:

        LayerReLU(std::shared_ptr<Batch<float>> input, std::shared_ptr<Batch<float>> output);

        void Forward() override;

        ~LayerReLU() override = default;
    };
}








#endif