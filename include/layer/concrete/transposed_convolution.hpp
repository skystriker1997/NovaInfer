#ifndef NOVA_INFER_LAYER_TRANSPOSED_CONVOLUTION
#define NOVA_INFER_LAYER_TRANSPOSED_CONVOLUTION


#include "layer/layer.hpp"

namespace nova_infer {
    class LayerTransposedConvolution : public Layer {
    private:

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        Batchf weights_;
        bool use_bias_;
        Eigen::RowVectorXf bias_;

        int groups_;
        int padding_h_;
        int padding_w_;
        int stride_h_;
        int stride_w_;

        int output_padding_h_;
        int output_padding_w_;

    public:

        LayerTransposedConvolution(
                std::string_view name,
                std::vector<std::string> input_name,
                std::vector<std::string> output_name,
                Batchf weights,
                bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf> bias,
                int padding_h, int padding_w,
                int stride_h, int stride_w,
                int groups,
                int output_padding_h, int output_padding_w
        );

        void AttachInput(const std::shared_ptr<Batchf> &input) override {input_ = input;};

        void AttachOutput(const std::shared_ptr<Batchf> &output) override {output_ = output;};

        void Forward() override;

        ~LayerTransposedConvolution() override = default;
    };

    std::shared_ptr<LayerTransposedConvolution> MakeLayerTransposedConvolution(pnnx::Operator *opt);


}








#endif