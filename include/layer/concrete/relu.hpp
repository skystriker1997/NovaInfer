#ifndef NOVA_INFER_LAYER_RELU
#define NOVA_INFER_LAYER_RELU


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerReLU: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

    public:

        LayerReLU(std::string name, std::vector<std::string> input_name, std::vector<std::string> output_name);

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerReLU() override = default;
    };

    std::shared_ptr<LayerReLU> MakeLayerReLU(pnnx::Operator *opt);
}








#endif