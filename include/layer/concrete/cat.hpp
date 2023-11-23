#ifndef NOVA_INFER_LAYER_CAT
#define NOVA_INFER_LAYER_CAT


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerCat: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_name_;

        std::vector<std::shared_ptr<Batchf>> inputs_;
        std::shared_ptr<Batchf> output_;

        int dim_;

    public:

        LayerCat(std::string name, std::vector<std::string> input_names, std::vector<std::string> output_name, int dim);

        void AssignInput(std::vector<std::shared_ptr<Batchf>> inputs) {inputs_ = inputs;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerCat() override = default;
    };

    std::shared_ptr<LayerCat> MakeLayerCat(pnnx::Operator *opt);
}



#endif