#ifndef NOVA_INFER_LAYER_FLATTEN
#define NOVA_INFER_LAYER_FLATTEN


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerFlatten: public Layer {
    private:

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int start_dim_;
        int end_dim_;

    public:

        LayerFlatten(std::string_view name,
                     std::vector<std::string> input_name, std::vector<std::string> output_name,
                     int start_dim, int end_dim);

        void AttachInput(const std::shared_ptr<Batchf> &input) override {input_ = input;};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerFlatten() override = default;
    };

    std::shared_ptr<LayerFlatten> MakeLayerFlatten(pnnx::Operator *opt);
}




#endif