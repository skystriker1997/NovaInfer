#include "layer/concrete/convolution.hpp"

namespace nova_infer {
    LayerConvolution::LayerConvolution(std::string_view name,
                                       std::vector<std::string> input_name, std::vector<std::string> output_name,
                                       Batchf weights,
                                       bool use_bias, const Eigen::Ref<const Eigen::RowVectorXf> bias,
                                       int padding_h, int padding_w,
                                       int stride_h, int stride_w,
                                       int groups
                                       )
        {
        check_(groups>=1) << "failed to construct layer convolution; number of groups should be a positive integer";
        check_(padding_h>=0 && padding_w>=0) << "failed to construct layer convolution; padding should be non-negative integers";
        check_(stride_h>=1 && stride_w>=1) << "failed to construct layer convolution; stride should be positive integers";

        type_ = LayerType::Conv;
        name_ = name;
        input_names_ = std::move(input_name);
        output_names_ = std::move(output_name);
        weights_ = std::move(weights);
        bias_ = bias;
        groups_ = groups;
        use_bias_ = use_bias;
        padding_h_ = padding_h;
        padding_w_ = padding_w;
        stride_h_ = stride_h;
        stride_w_ = stride_w;
    };



    void LayerConvolution::Forward() {

        check_(input_->size() == output_->size()) << "failed to execute layer convolution; input and output should have same batch size";

        int batch_size = input_->size();
        int kernels = weights_.size();

        int kernel_h = weights_[0].Rows();
        int kernel_w = weights_[0].Cols();
        int kernel_c = weights_[0].Channels();

        omp_set_num_threads(omp_get_num_procs());

        for(int t=0; t<batch_size; t++) {
            Tensor<float> &in = input_->at(t);
            Tensor<float> &out = output_->at(t);
            check_(kernels==out.Channels()) << "failed to execute layer convolution; the number of output channels is mismatching with that of kernels";
            check_(in.Channels() % groups_ == 0) << "failed to execute convolution; the number of input channels should be divisible by number of groups";
            check_(in.Channels()/groups_ == kernel_c) << "failed to execute layer convolution; the number of channels per group of input is not matching with that of a kernel";
            int output_height = ((in.Rows() + 2*padding_h_) - kernel_h) / stride_h_ + 1;
            int output_width = ((in.Cols() + 2*padding_w_) - kernel_w) / stride_w_ + 1;
            check_(out.Rows() == output_height) << "failed to execute convolution; incorrect row number of output";
            check_(out.Cols() == output_width) << "failed to execute convolution; incorrect col number of output";

            int kernels_per_group = kernels / groups_;

            for(int g=0; g<groups_; g++) {
                int start_channel = g*kernel_c;
                Tensor<float> in_padding = in.Padding({padding_h_, padding_w_, padding_h_, padding_w_}, 0);
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> im2col_input = Im2ColInput(start_channel, kernel_c, in_padding, output_height, output_width, kernel_h, kernel_w);
                int start_kernel = kernels_per_group*g;
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> im2col_weights = Im2ColWeights(start_kernel, kernels_per_group);
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result = (im2col_input * im2col_weights.transpose()).rowwise() + bias_.segment(start_kernel, kernels_per_group);
                for(int k=0; k<kernels_per_group; k++) {
                    out.WriteMatrix(start_kernel+k) = result.col(k).template reshaped<Eigen::RowMajor> (output_height, output_width);
                }
            }
        }
    }




    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> LayerConvolution::Im2ColWeights(int begin, int n) {
        check_(begin>=0 && begin+n-1<weights_.size()) << "failed to Im2Col weights; the begin and end index cannot be out of kernel range";
        int h = n;
        int w = weights_[begin].Channels() * weights_[begin].Rows() * weights_[begin].Cols();

        int kernel_c = weights_[begin].Channels();
        int kernel_h = weights_[begin].Rows();
        int kernel_w = weights_[begin].Cols();
        int kernel_hw = kernel_h*kernel_w;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(h, w);
        for(int r=0; r<h; r++) {
            for(int c=0; c<kernel_c; c++) {
                result.block(r, c*kernel_hw, 1, kernel_hw) = weights_[begin+r].ReadMatrix(c).template reshaped<Eigen::RowMajor>(1, kernel_hw);
            }
        }
        return result;
    }


    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> LayerConvolution::Im2ColInput(int begin, int n, Tensor<float>& input, int steps_h, int steps_w, int kernel_h, int kernel_w) {
        check_(begin>=0 && begin+n-1<input.Channels()) << "failed to Im2Col input; the begin and end index cannot be out of channel range of input";

        int height = steps_h * steps_w;
        int width = kernel_h * kernel_w * n;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(height, width);

        for(int h=0; h<steps_h; h++) {
            for(int w=0; w<steps_w; w++) {
                for(int c=begin; c<begin+n; c++) {
                    result.block(h*steps_w+w, (c-begin)*kernel_h*kernel_w, 1, kernel_h*kernel_w) = input.ReadMatrix(c).block(stride_h_*h, stride_w_*w, kernel_h, kernel_w).template reshaped<Eigen::RowMajor>(1, kernel_h*kernel_w);
                }
            }
        }
        return result;
    }


    std::shared_ptr<LayerConvolution> MakeLayerConvolution(pnnx::Operator *opt) {
        Check check;

        check(opt->inputs.size()==1) << "failed to create layer convolution; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer convolution; only produce one tensor as output";

        auto groups = opt->params.find("groups");
        check(groups != opt->params.end()) << "failed to create layer convolution; cannot find parameter groups";

        auto dilation = opt->params.find("dilation");
        check(dilation != opt->params.end()) << "failed to create layer convolution; cannot find parameter dilation";
        check(dilation->second.ai.size()==2) << "failed to create layer convolution; parameter dilation should have two elements";

        auto in_channels = opt->params.find("in_channels");
        check(in_channels != opt->params.end()) << "failed to create layer convolution; cannot find parameter in_channels";

        auto out_channels = opt->params.find("out_channels");
        check(out_channels != opt->params.end()) << "failed to create layer convolution; cannot find parameter out_channels";

        auto padding = opt->params.find("padding");
        check(padding != opt->params.end()) << "failed to create layer convolution; cannot find parameter padding";
        check(padding->second.ai.size()==2) << "failed to create layer convolution; parameter padding should have two elements";

        auto use_bias = opt->params.find("bias");
        check(use_bias != opt->params.end()) << "failed to create layer convolution; cannot find parameter bias";

        auto stride = opt->params.find("stride");
        check(stride != opt->params.end()) << "failed to create layer convolution; cannot find parameter stride";
        check(stride->second.ai.size()==2) << "failed to create layer convolution; parameter stride should have two elements";

        auto kernel_size = opt->params.find("kernel_size");
        check(kernel_size != opt->params.end()) << "failed to create layer convolution; cannot find parameter kernel_size";
        check(kernel_size->second.ai.size()==2) << "failed to create layer convolution; parameter kernel_size should have two elements";

        auto weights = opt->attrs.find("weight");
        check(weights != opt->attrs.end()) << "failed to create layer convolution; cannot find attribute weights";

        int kernels = out_channels->second.i;

        int channels_input = in_channels->second.i;

        int num_groups = groups->second.i;

        check(channels_input % num_groups == 0) << "failed to create layer convolution; number of channels in input tensor should be divisible by number of groups";
        int kernel_c = channels_input / num_groups;

        int kernel_h = kernel_size->second.ai[0];
        int kernel_w = kernel_size->second.ai[1];

        int dilation_h = dilation->second.ai[0];
        int dilation_w = dilation->second.ai[1];

        std::vector<int> tensor_shape = {kernel_c, kernel_h, kernel_w};

        Batchf weights_f(kernels, Tensor<float>(tensor_shape));

        int non_empty_h = dilation_h > 1 ? (kernel_h+dilation_h-1)/dilation_h : kernel_h;
        int non_empty_w = dilation_w > 1 ? (kernel_w+dilation_w-1)/dilation_w : kernel_w;

        auto convert_to_float = [](std::vector<char> &attr_val, Check &check){
            std::vector<float> vect_float;
            auto float_size = sizeof(float);
            check(attr_val.size() % float_size == 0) << "failed to convert char arr to float arr; total bytes should be divisible by size of a float";
            for(auto i=0; i<attr_val.size()/float_size; i++) {
                float f = *((float*)attr_val.data()+i);
                vect_float.emplace_back(f);
            }
            return vect_float;
        };

        std::vector<float> weights_converted = convert_to_float(weights->second.data, check);

        check(kernels*kernel_c*non_empty_h*non_empty_w == weights_converted.size()) << "failed to create layer convolution; kernel shape is mismatching with number of elements";

        const int kernel_hw = non_empty_w * non_empty_h;

        check(kernels % num_groups == 0) << "failed to create layer convolution; number of kernels should be divisible by number of groups";
        int kernels_per_group = kernels / num_groups;

        const int kernel_nhw = kernel_hw * kernels_per_group;
        const int kernel_cnhw = kernel_nhw * kernel_c;

        for(int g=0; g<num_groups; g++) {
            std::vector<float> sub_weights = {weights_converted.begin()+g*kernel_cnhw, weights_converted.begin()+(g+1)*kernel_cnhw};
            for(int n=0; n<kernels_per_group; n++) {
                int offset_wrt_kernel = n*kernel_hw;   // offset with respect to kernel
                for(int c=0; c<kernel_c; c++) {
                    int offset_wrt_channel = c*kernel_nhw;
                    for(int h=0; h<non_empty_h; h++) {
                        int offset_wrt_height = h*non_empty_w;
                        for(int w=0; w<non_empty_w; w++) {
                            int h_index = h*dilation_h;
                            int w_index = w*dilation_w;
                            weights_f[kernels_per_group*g+n].WriteMatrix(c)(h_index, w_index) = sub_weights[offset_wrt_channel+offset_wrt_kernel+offset_wrt_height+w];
                        }
                    }
                }
            }
        }

        Eigen::RowVectorXf bias_f;

        if(use_bias->second.b) {
            auto bias = opt->attrs.find("bias");
            check(bias != opt->attrs.end()) << "failed to create layer convolution; cannot find attribute bias";

            std::vector<float> bias_farr = convert_to_float(bias->second.data, check);

            bias_f = Eigen::Map<Eigen::RowVectorXf>(bias_farr.data(), bias_farr.size());
        }

        std::vector<std::string> input_name = {opt->inputs[0]->name};
        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerConvolution>(opt->name,
                                                  std::move(input_name), std::move(output_name),
                                                  std::move(weights_f),
                                                  use_bias->second.b, bias_f,
                                                  padding->second.ai[0],padding->second.ai[1],
                                                  stride->second.ai[0], stride->second.ai[1],
                                                  groups->second.i);
    };



}