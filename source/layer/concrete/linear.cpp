#include "layer/concrete/linear.hpp"


namespace sky_infer {
    LayerLinear::LayerLinear(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output, const Eigen::Ref<const Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& weights, bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf>& bias): type_(LayerType::Linear), name_(std::move(name)), input_(std::move(input)), output_(std::move(output)), weights_(weights), use_bias(use_bias), bias_(bias) {}


    void LayerLinear::Forward() {

        check_(input_->size() == output_->size()) << "failed to execute linear; input and output have different batch sizes";

        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_t = weights_.transpose();

        if(use_bias) {
            for(int i=0; i<input_->size(); i++) {
                Tensor<float>& in = input_->at(i);
                Tensor<float>& out = output_->at(i);
                check_(in.Channels()==1 && out.Channels()==1) << "failed to execute linear; each tensor is expected to have only one channel after been flattened";
                int in_samples = in.Rows();
                int in_features = in.Cols();
                int out_samples = out.Rows();
                int out_features = out.Cols();
                check_(weights_.cols()==in_features) << "failed to execute linear; the size of each weights vector should be matching with that of each sample of input features";
                check_(weights_.rows()==out_features) << "failed to execute linear; the number of weights vectors should be matching with the size of each sample of output features";
                check_(in_samples==out_samples) << "failed to execute linear; number of samples should be constant during execution";
                check_(bias_.size() == out_features) << "failed to execute linear; the size of bias vector should be same with that of each sample of output features";

                out.WriteMatrix(0) = (in.ReadMatrix(0) * weights_t).rowwise() + bias_;
            }
        } else {
            for(int i=0; i<input_->size(); i++) {
                Tensor<float>& in = input_->at(i);
                Tensor<float>& out = output_->at(i);
                check_(in.Channels()==1 && out.Channels()==1) << "failed to execute linear; each tensor is expected to have only one channel after been flattened";
                int in_samples = in.Rows();
                int in_features = in.Cols();
                int out_samples = out.Rows();
                int out_features = out.Cols();
                check_(weights_.cols()==in_features) << "failed to execute linear; the size of each weights vector should be matching with that of each sample of input features";
                check_(weights_.rows()==out_features) << "failed to execute linear; the number of weights vectors should be matching with the size of each sample of output features";
                check_(in_samples==out_samples) << "failed to execute linear; number of samples should be constant during execution";

                out.WriteMatrix(0) = in.ReadMatrix(0) * weights_t;
            }
        }
    }

}