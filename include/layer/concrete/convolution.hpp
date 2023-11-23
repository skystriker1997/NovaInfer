#ifndef NOVA_INFER_LAYER_CONVOLUTION
#define NOVA_INFER_LAYER_CONVOLUTION


#include "layer/layer.hpp"


namespace nova_infer {
    class LayerConvolution : public Layer {
    private:

        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

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


        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Im2ColWeights(int begin, int end);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Im2ColInput(int begin, int end, Tensor<float> &input, int steps_h, int steps_w, int kernel_h, int kernel_w);



    public:

        LayerConvolution(std::string name,
                         std::vector<std::string> input_name,
                         std::vector<std::string> output_name,
                         Batchf weights,
                         bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf> bias,
                         int padding_h, int padding_w,
                         int stride_h, int stride_w,
                         int groups
                         );

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerConvolution() override = default;
    };


    std::shared_ptr<LayerConvolution> MakeLayerConvolution(pnnx::Operator *opt);


}








#endif