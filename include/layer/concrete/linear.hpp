#ifndef SKY_INFER_LAYER_LINEAR
#define SKY_INFER_LAYER_LINEAR


#include "layer/layer.hpp"

namespace sky_infer {
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
        bool use_bias;
        Eigen::RowVectorXf bias_;


    public:

        LayerLinear(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output, const Eigen::Ref<const Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& weights, bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf>& bias);

        void Forward() override;

        ~LayerLinear() override = default;
    };
}




#endif