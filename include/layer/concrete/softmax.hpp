#ifndef NOVA_INFER_LAYER_SOFTMAX
#define NOVA_INFER_LAYER_SOFTMAX


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerSoftmax: public Layer {
    private:

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int dim_;


    public:

        LayerSoftmax(std::string_view name, std::vector<std::string> input_name, std::vector<std::string> output_name, int dim);

        void AttachInput(const std::shared_ptr<Batchf> &input) override {input_ = input;};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerSoftmax() override = default;
    };

    std::shared_ptr<LayerSoftmax> MakeLayerSoftmax(pnnx::Operator *opt);
}




#endif