#ifndef SKY_INFER_OPERAND
#define SKY_INFER_OPERAND


#include "type.hpp"
#include <glog/logging.h>
#include <string>
#include <tensor/tensor.hpp>
#include "pnnx/ir.h"

namespace sky_infer {

    class Operand {
    public:
        std::string  name_;
        std::vector<int> shape_;
        std::vector<Tensor<float>> data_;
        OpdDataType data_type_;

        Operand(pnnx::Operand* pnnx_opd);
        ~Operand() = default;

    };


}


#endif