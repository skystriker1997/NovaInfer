#ifndef SKY_INFER_PARAMETER
#define SKY_INFER_PARAMETER

#include <memory>
#include <vector>
#include <string>
#include "type.hpp"


namespace sky_infer {
    struct Parameter {
        ParamDataType type_;
      //  ParamDataType value_;
        Parameter() : type_(ParamDataType::ParamDummy) {};
        virtual ~Parameter() = default;
    };


    struct ParameterBool : public Parameter {
        ParamDataType type_;
        bool value_;
        ParameterBool() : type_(ParamDataType::ParamBool) {};
        explicit ParameterBool(bool value) : type_(ParamDataType::ParamBool), value_(value) {};
        ~ParameterBool() override = default;
    };


    struct ParameterInt : public Parameter {
        ParamDataType type_;
        int value_;
        ParameterInt() : type_(ParamDataType::ParamInt) {};
        explicit ParameterInt(int value) : type_(ParamDataType::ParamInt), value_(value) {};
        ~ParameterInt() override = default;
    };


    struct ParameterFloat : public Parameter {
        ParamDataType type_;
        float value_;
        ParameterFloat() : type_(ParamDataType::ParamFloat) {};
        explicit ParameterFloat(float value) : type_(ParamDataType::ParamFloat), value_(value) {};
        ~ParameterFloat() override = default;
    };


    struct ParameterString : public Parameter {
        ParamDataType type_;
        std::string value_;
        ParameterString() : type_(ParamDataType::ParamString) {};
        explicit ParameterString(std::string value) : type_(ParamDataType::ParamString), value_(value) {};
        ~ParameterString() override = default;
    };


    struct ParameterIntArray : public Parameter {
        ParamDataType type_;
        std::vector<int> value_;
        ParameterIntArray() : type_(ParamDataType::ParamIntArray) {};
        explicit ParameterIntArray(std::vector<int> value) : type_(ParamDataType::ParamIntArray), value_(value) {};
        ~ParameterIntArray() override = default;
    };


    struct ParameterFloatArray : public Parameter {
        ParamDataType type_;
        std::vector<float> value_;
        ParameterFloatArray() : type_(ParamDataType::ParamFloatArray) {};
        explicit ParameterFloatArray(std::vector<float> value) : type_(ParamDataType::ParamFloatArray), value_(value) {};
        ~ParameterFloatArray() override = default;
    };


    struct ParameterStringArray : public Parameter {
        ParamDataType type_;
        std::vector<std::string> value_;
        ParameterStringArray() : type_(ParamDataType::ParamStringArray) {};
        explicit ParameterStringArray(std::vector<std::string> value) : type_(ParamDataType::ParamStringArray), value_(value) {};
        ~ParameterStringArray() override = default;
    };

}


#endif