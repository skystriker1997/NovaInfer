#ifndef SKY_INFER_LAYER_INPUT
#define SKY_INFER_LAYER_INPUT


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerInput: public Layer {
    public:

        LayerType type_;

        LayerInput(): type_(LayerType::Input) {};


        ~LayerInput() override = default;
    };
}








#endif