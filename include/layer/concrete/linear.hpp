#ifndef NOVA_INFER_LAYER_LINEAR
#define NOVA_INFER_LAYER_LINEAR


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerLinear: public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;


        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_;
        bool use_bias_;
        Eigen::RowVectorXf bias_;


    public:

        LayerLinear(std::string name,
                    std::vector<std::string> input_name, std::vector<std::string> output_name,
                    const Eigen::Ref<const Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights,
                    bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf> bias);

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerLinear() override = default;
    };

    std::shared_ptr<LayerLinear> MakeLayerLinear(pnnx::Operator *opt);
}




#endif