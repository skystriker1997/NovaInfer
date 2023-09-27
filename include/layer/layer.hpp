#ifndef SKY_INFER_LAYER
#define SKY_INFER_LAYER


#include "type.hpp"
#include "tensor/tensor.hpp"


namespace sky_infer {

    class Graph;


    class Layer {
    private:
        std::string name_;
        LayerType type_;
        std::vector<std::shared_ptr<Batch<float>>> inputs_;
        std::shared_ptr<Batch<float>> output_;


    public:

        Layer(): type_(LayerType::Dummy) {};

        std::vector<std::shared_ptr<Batch<float>>> GetInputs() {return inputs_;};
        std::shared_ptr<Batch<float>> GetOutput() {return output_;};
        std::string GetName() {return name_;};
        LayerType GetType() {return type_;};

        virtual void Forward() {};

        virtual ~Layer() = default;

    };


}


#endif