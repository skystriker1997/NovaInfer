#ifndef SKY_INFER_OPERATOR
#define SKY_INFER_OPERATOR


#include <glog/logging.h>
#include "type.hpp"
#include "operand.hpp"
#include "parameter.hpp"
#include "attribute.hpp"
#include "tensor/tensor.hpp"
#include "vector"
#include "map"
#include <memory>
#include "pnnx/ir.h"





namespace sky_infer {


    class Layer;

    class Operator {
    public:


        std::string name_;

        std::vector<Operand*> inputs_;
        std::vector<Operand*> outputs_;

        std::map<std::string, Attribute> attributes_;
        std::map<std::string, std::unique_ptr<Parameter>> params_;
        std::string type_;


        Layer* layer_;

        explicit Operator(const pnnx::Operator* pnnx_opt);


        ~Operator() = default;

    };



}


#endif