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
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

    public:

        Layer(): type_(LayerType::Dummy) {};

//        std::shared_ptr<Batchf> GetInput() {return input_;};
//        std::shared_ptr<Batchf> GetOutput() {return output_;};

     //   long foo() {return inputs_.size();};

        std::string GetName() {return name_;};
        const std::vector<std::string>& GetInputName() {return input_name_;};
        const std::vector<std::string>& GetOutputName() {return output_name_;};

        LayerType GetType() {return type_;};

        virtual void Forward() {};

        virtual ~Layer() = default;

    };


}


#endif