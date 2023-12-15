#ifndef NOVA_INFER_LAYER_RELU
#define NOVA_INFER_LAYER_RELU


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerReLU: public Layer {
    private:

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

    public:

        LayerReLU(std::string_view name, std::vector<std::string> input_name, std::vector<std::string> output_name);

        void AttachInput(const std::shared_ptr<Batchf> &input) override {input_ = input;};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerReLU() override = default;
    };

    std::shared_ptr<LayerReLU> MakeLayerReLU(pnnx::Operator *opt);
}








#endif