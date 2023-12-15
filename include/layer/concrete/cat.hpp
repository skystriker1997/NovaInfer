#ifndef NOVA_INFER_LAYER_CAT
#define NOVA_INFER_LAYER_CAT


#include "layer/layer.hpp"

namespace nova_infer {

    class LayerCat: public Layer {

    private:
        std::vector<std::shared_ptr<Batchf>> inputs_;
        std::shared_ptr<Batchf> output_;

        int dim_;

    public:
        LayerCat(std::string_view name, std::vector<std::string> input_names, std::vector<std::string> output_name, int dim);

        void AttachInput(const std::shared_ptr<Batchf> &input) override {inputs_.emplace_back(input);};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerCat() override = default;
    };

    std::shared_ptr<LayerCat> MakeLayerCat(pnnx::Operator *opt);
}



#endif