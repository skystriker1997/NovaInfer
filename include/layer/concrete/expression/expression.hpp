#ifndef SKY_INFER_LAYER_EXPRESSION
#define SKY_INFER_LAYER_EXPRESSION

#include "layer/layer.hpp"
#include "layer/concrete/expression/parser.hpp"
#include "tensor/tensor.hpp"


namespace sky_infer {
    class LayerExpression: public Layer {
    public:
        LayerType type_;

        LayerExpression(): type_(LayerType::Expression) {};

        void Forward(Operator* opt) override;

        ~LayerExpression() override = default;
    };
}




#endif