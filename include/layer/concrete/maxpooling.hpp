#ifndef SKY_INFER_LAYER_MAXPOOLING
#define SKY_INFER_LAYER_MAXPOOLING


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerMaxpooling: public Layer {
    public:
        LayerType type_;
        LayerMaxpooling(): type_(LayerType::MaxPooling) {};
        void Forward(Operator* opt) override;
        ~LayerMaxpooling() override;
    };
}





#endif