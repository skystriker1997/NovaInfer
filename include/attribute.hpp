#ifndef SKY_INFER_ATTRIBUTE
#define SKY_INFER_ATTRIBUTE

#include <memory>
#include <vector>
#include "type.hpp"
#include <glog/logging.h>
#include <type_traits>
#include "tensor/tensor.hpp"
#include "pnnx/ir.h"

namespace sky_infer {

    class Attribute {
    private:
        std::vector<Tensor<float>> data_;
        std::vector<int> shape_;
        AttrDataType type_;

    public:
        explicit Attribute(const pnnx::Attribute& pnnx_attr);
        ~Attribute() = default;

    };


}


#endif