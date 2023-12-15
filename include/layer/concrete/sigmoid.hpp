#ifndef NONA_INFER_LAYER_SIGMOID
#define NONA_INFER_LAYER_SIGMOID


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerSigmoid: public Layer {
    private:

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;


    public:

        LayerSigmoid(std::string_view name, std::vector<std::string> input_name, std::vector<std::string> output_name);

        void AttachInput(const std::shared_ptr<Batchf> &input) override {input_ = input;};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerSigmoid() override = default;
    };

    std::shared_ptr<LayerSigmoid> MakeLayerSigmoid(pnnx::Operator *opt);
}




#endif