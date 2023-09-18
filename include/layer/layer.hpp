#ifndef SKY_INFER_LAYER
#define SKY_INFER_LAYER


#include "type.hpp"
#include <glog/logging.h>
#include "tensor/tensor.hpp"
#include "operand.hpp"
#include "operator.hpp"


namespace sky_infer {

    class Layer {
    public:
        LayerType type_;

        Layer(): type_(LayerType::Dummy) {};

        virtual void Forward(Operator* opt) {};

        virtual ~Layer() = default;

    };


}


#endif