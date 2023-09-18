#ifndef SKY_INFER_LAYER_RELU
#define SKY_INFER_LAYER_RELU


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerReLU: public Layer {
    public:

        LayerType type_;


        LayerReLU(): type_(LayerType::ReLU) {};


        void Forward(Operator *opd) override;

        ~LayerReLU() override = default;
    };
}








#endif