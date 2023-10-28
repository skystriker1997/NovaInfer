#ifndef SKY_INFER_LAYER_FLATTEN
#define SKY_INFER_LAYER_FLATTEN


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerFlatten: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int start_dim_;
        int end_dim_;

    public:

        LayerFlatten(std::string name,
                     std::vector<std::string> input_name, std::vector<std::string> output_name,
                     int start_dim, int end_dim);

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerFlatten() override = default;
    };

    std::shared_ptr<LayerFlatten> MakeLayerFlatten(pnnx::Operator *opt);
}




#endif