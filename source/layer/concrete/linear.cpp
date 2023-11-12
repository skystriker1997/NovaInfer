#include "layer/concrete/linear.hpp"


namespace sky_infer {
    LayerLinear::LayerLinear(std::string name,
                             std::vector<std::string> input_name, std::vector<std::string> output_name,
                             const Eigen::Ref<const Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &weights,
                             bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf> &bias)
    {
        type_ = LayerType::Linear;
        name_ = std::move(name);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
        weights_ = weights;
        use_bias_ = use_bias;
        bias_ = bias;
    }


    void LayerLinear::Forward() {

        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_t = weights_.transpose();

        if(use_bias_) {
            omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for default(shared)
            for(int i=0; i<input_->size(); i++) {
                Tensor<float> &in = input_->at(i);
                Tensor<float> &out = output_->at(i);
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
            omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for default(shared)
            for(int i=0; i<input_->size(); i++) {
                Tensor<float> &in = input_->at(i);
                Tensor<float> &out = output_->at(i);
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

    std::shared_ptr<LayerLinear> MakeLayerLinear(pnnx::Operator *opt) {
        Check check;
        check(opt->inputnames.size()==1) << "failed to create layer linear; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer linear; only produce one tensor as output";

        auto use_bias = opt->params.find("bias");
        check(use_bias != opt->params.end()) << "failed to create linear layer; cannot find parameter bias";
        Eigen::RowVectorXf bias_f;

        auto convert_to_float = [](std::vector<char>& attr_val, Check& check){
            std::vector<float> vect_float;
            auto float_size = sizeof(float);
            check(attr_val.size() % float_size == 0) << "failed to convert char arr to float arr; total bytes cannot be divided by size of a float";
            for(auto i=0; i<attr_val.size()/float_size; i++) {
                float f = *((float*)attr_val.data()+i);
                vect_float.emplace_back(f);
            }
            return vect_float;
        };

        if(use_bias->second.b) {
            auto bias = opt->attrs.find("bias");
            check(bias != opt->attrs.end()) << "failed to create linear layer; cannot find attribute bias";

            std::vector<float> bias_converted = convert_to_float(bias->second.data, check);

            bias_f = Eigen::Map<Eigen::RowVectorXf>(bias_converted.data(), bias_converted.size());
        }

        auto weights = opt->attrs.find("weight");
        check(weights != opt->attrs.end()) << "failed to create linear layer; cannot find attribute weight";

        std::vector<int>& shape = weights->second.shape;
        check(!shape.empty()) << "failed to create linear layer; weights should be 2-dimensional";
        if(shape.size()>2) {
            for(int i=0; i<shape.size()-2; i++)
                check(shape[i]==1) << "failed to create linear layer; weights should be 2-dimensional";
        }
        int cols = shape.back();
        int rows = shape.size()==1?1:shape[shape.size()-2];

        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_f = Eigen::Map<Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(convert_to_float(weights->second.data, check).data(), rows, cols);

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerLinear>(std::move(opt->name), std::move(opt->inputnames), std::move(output_name), weights_f, use_bias->second.b, bias_f);
    };

}