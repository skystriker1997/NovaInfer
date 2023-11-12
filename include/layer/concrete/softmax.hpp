#ifndef SKY_INFER_LAYER_SOFTMAX
#define SKY_INFER_LAYER_SOFTMAX


#include "layer/layer.hpp"

namespace sky_infer {
    class LayerSoftmax: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int dim_;


    public:

        LayerSoftmax(std::string name, std::vector<std::string> input_name, std::vector<std::string> output_name, int dim);

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerSoftmax() override = default;
    };

    std::shared_ptr<LayerSoftmax> MakeLayerSoftmax(pnnx::Operator *opt);
}




#endif