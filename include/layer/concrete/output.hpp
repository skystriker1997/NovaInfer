#ifndef SKY_INFER_LAYER_OUTPUT
#define SKY_INFER_LAYER_OUTPUT


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerOutput: public Layer {
    public:

        LayerType type_;

        LayerOutput(): type_(LayerType::Output) {};


        ~LayerOutput() override = default;
    };
}



#endif