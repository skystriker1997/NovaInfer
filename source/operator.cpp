#include "operator.hpp"



namespace sky_infer {
    Operator::Operator(const pnnx::Operator* pnnx_opt) {
        CHECK(pnnx_opt) << "failed to construct operator; invalid pnnx operator pointer";

        name_ = pnnx_opt->name;
        type_ = pnnx_opt->type;
        layer_ = nullptr;

        // initialise parameters
        for (const auto &[name, pnnx_param]: pnnx_opt->params) {
            const int type = pnnx_param.type;
            switch (type) {
//                case (int(ParamDataType::ParamUnknown)): {
//                    params_.insert({name, new Parameter()});
//                    break;
//                }
                case (int(ParamDataType::ParamBool)): {
                    params_.insert({name, std::make_unique<ParameterBool>(pnnx_param.b)});
                    break;
                }
                case (int(ParamDataType::ParamInt)): {
                    params_.insert({name, std::make_unique<ParameterInt>(pnnx_param.i)});
                    break;
                }
                case (int(ParamDataType::ParamFloat)): {
                    params_.insert({name, std::make_unique<ParameterFloat>(pnnx_param.f)});
                    break;
                }
                case (int(ParamDataType::ParamString)): {
                    params_.insert({name, std::make_unique<ParameterString>(pnnx_param.s)});
                    break;
                }
                case (int(ParamDataType::ParamIntArray)): {
                    params_.insert({name, std::make_unique<ParameterIntArray>(pnnx_param.ai)});
                    break;
                }
                case (int(ParamDataType::ParamFloatArray)): {
                    params_.insert({name, std::make_unique<ParameterFloatArray>(pnnx_param.af)});
                    break;
                }
                case (int(ParamDataType::ParamStringArray)): {
                    params_.insert({name, std::make_unique<ParameterStringArray>(pnnx_param.as)});
                    break;
                }
                default: {
                    LOG(FATAL) << "failed to construct operator; unknown data type of parameter: " << type;
                }
            }
        }

        // initialise attributes
        for (const auto &[name, pnnx_attr]: pnnx_opt->attrs) {
            switch (pnnx_attr.type) {
                case 1: {
                    attributes_.insert({name, Attribute(pnnx_attr)});
                    break;
                }
                default: {
                    LOG(FATAL) << "failed to construct operator; unsupported attribute data type: " << pnnx_attr.type;
                }
            }
        }

        // initialise output Operands
//        for (pnnx::Operand *pnnx_opd: pnnx_opt->outputs)
//            outputs_.emplace_back(new Operand(pnnx_opd));

    }

//    void Operator::Execute() {
//        layer_->;
//    }


//    Operator::~Operator() {
//        for(const auto &[name, param] : params_)
//            delete param;
//    }
}