#ifndef NOVA_INFER_LAYER
#define NOVA_INFER_LAYER


#include "type.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include "pnnx/ir.h"
#include <omp.h>


namespace nova_infer {

    class Graph;


    class Layer {
    private:
        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

    public:

        Layer(): type_(LayerType::Dummy) {};

        std::string GetName() {return name_;};
        const std::vector<std::string> &GetInputName() {return input_names_;};
        const std::vector<std::string> &GetOutputName() {return output_names_;};

        LayerType GetType() {return type_;};

        virtual void AssignInput(std::shared_ptr<Batchf> input) {};

        virtual void AssignOutput(std::shared_ptr<Batchf> output) {};

        virtual void Forward() {};

        virtual ~Layer() = default;

    };


}


#endif