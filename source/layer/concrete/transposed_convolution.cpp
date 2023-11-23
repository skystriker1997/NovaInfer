#include "layer/concrete/transposed_convolution.hpp"


namespace nova_infer {
    LayerTransposedConvolution::LayerTransposedConvolution(std::string name, std::vector<std::string> input_name,
                                                           std::vector<std::string> output_name,
                                                           Batchf weights,
                                                           bool use_bias,
                                                           const Eigen::Ref<const Eigen::RowVectorXf> bias,
                                                           int padding_h, int padding_w, int stride_h, int stride_w,
                                                           int groups, int output_padding_h, int output_padding_w) {
        check_(groups>=1) << "failed to construct layer transposed convolution; number of groups should be a positive integer";
        check_(padding_h>=0 && padding_w>=0) << "failed to construct layer transposed convolution; padding should be non-negative integers";
        check_(stride_h>=1 && stride_w>=1) << "failed to construct layer transposed convolution; stride should be positive integers";
        check_(output_padding_h>=0 && output_padding_w>=0) << "failed to construct layer transposed convolution; padding of the output should be zero";
        type_ = LayerType::DeConv;
        name_ = std::move(name);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
        weights_ = std::move(weights);
        bias_ = bias;
        groups_ = groups;
        use_bias_ = use_bias;
        padding_h_ = padding_h;
        padding_w_ = padding_w;
        stride_h_ = stride_h;
        stride_w_ = stride_w;
        output_padding_h_ = output_padding_h;
        output_padding_w_ = output_padding_w;
    }

    void LayerTransposedConvolution::Forward() {

        int batch_size = input_->size();
        int kernels = weights_.size();

        int kernel_h = weights_[0].Rows();
        int kernel_w = weights_[0].Cols();
        int kernel_c = weights_[0].Channels();

        omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
        for(int t=0; t<batch_size; t++) {
            Tensor<float> &in = input_->at(t);
            Tensor<float> &out = output_->at(t);
            check_(kernels==out.Channels()) <<
            "failed to execute layer transposed convolution; the number of output channels is mismatching with that of kernels";
            check_(in.Channels() % groups_ == 0) <<
            "failed to execute layer transposed convolution; the number of input channels should be divisible by number of groups";
            check_(in.Channels()/groups_ == kernel_c) <<
            "failed to execute layer transposed convolution; the number of channels per group of input is not matching with that of a kernel";
            int output_height = (in.Rows() - 1) * stride_h_ + kernel_h + output_padding_h_ - 2*padding_h_;
            int output_width = (in.Cols() - 1) * stride_w_ + kernel_w + output_padding_w_ - 2*padding_w_;

            check_(out.Rows() == output_height) << "failed to execute layer transposed convolution; incorrect row number of output";
            check_(out.Cols() == output_width) << "failed to execute layer transposed convolution; incorrect col number of output";

            int kernels_per_group = kernels / groups_;

            Tensor<float> tmp = out.Padding({padding_h_, padding_w_, padding_h_, padding_w_}, 0);

            for(int g=0; g<groups_; g++) {
                int start_channel = g*kernel_c;
                int start_kernel = kernels_per_group*g;
                for(int k=0; k<kernels_per_group; k++) {
                    for(int h=0; h<in.Rows(); h++) {
                        for(int w=0; w<in.Cols(); w++) {
                            for(int c=0; c<kernel_c; c++) {
                                tmp.WriteMatrix(start_kernel+k).block(h*stride_h_, w*stride_w_, kernel_h, kernel_w) +=
                                        weights_[start_kernel+k].ReadMatrix(c) * in.ReadMatrix(start_channel+c)(h, w);
                            }
                        }
                    }
                    out.WriteMatrix(start_kernel+k) = tmp.ReadMatrix(start_kernel+k).block(padding_h_, padding_w_, output_height, output_width);
                    out.WriteMatrix(start_kernel+k).array() += bias_(start_kernel+k);
                }
            }
        }
    }

    std::shared_ptr<LayerTransposedConvolution> MakeLayerTransposedConvolution(pnnx::Operator *opt) {
        Check check;

        check(opt->inputnames.size()==1) << "failed to create layer transposed convolution; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer transposed convolution; only produce one tensor as output";

        auto groups = opt->params.find("groups");
        check(groups != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter groups";

        auto dilation = opt->params.find("dilation");
        check(dilation != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter dilation";
        check(dilation->second.ai.size()==2) << "failed to create layer transposed convolution; parameter dilation should have two elements";

        auto in_channels = opt->params.find("in_channels");
        check(in_channels != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter in_channels";

        auto out_channels = opt->params.find("out_channels");
        check(out_channels != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter out_channels";

        auto padding = opt->params.find("padding");
        check(padding != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter padding";
        check(padding->second.ai.size()==2) << "failed to create layer transposed convolution; parameter padding should have two elements";

        auto use_bias = opt->params.find("bias");
        check(use_bias != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter bias";

        auto stride = opt->params.find("stride");
        check(stride != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter stride";
        check(stride->second.ai.size()==2) << "failed to create layer transposed convolution; parameter stride should have two elements";

        auto kernel_size = opt->params.find("kernel_size");
        check(kernel_size != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter kernel_size";
        check(kernel_size->second.ai.size()==2) << "failed to create layer transposed convolution; parameter kernel_size should have two elements";

        auto output_padding = opt->params.find("output_padding");
        check(output_padding != opt->params.end()) << "failed to create layer transposed convolution; cannot find parameter output_padding";
        check(output_padding->second.ai.size()==2) << "failed to create layer transposed convolution; parameter output_padding should have two elements";


        auto weights = opt->attrs.find("weight");
        check(weights != opt->attrs.end()) << "failed to create layer transposed convolution; cannot find attribute weights";

        int kernels = out_channels->second.i;

        int channels_input = in_channels->second.i;

        int num_groups = groups->second.i;

        check(channels_input % num_groups == 0) << "failed to create layer transposed convolution; number of channels in input tensor should be divisible by number of groups";
        int kernel_c = channels_input / num_groups;


        int kernel_h = kernel_size->second.ai[0];
        int kernel_w = kernel_size->second.ai[1];

        int dilation_h = dilation->second.ai[0];
        int dilation_w = dilation->second.ai[1];

        if(dilation_h > 1)
            kernel_h += (kernel_h-1) * (dilation_h-1);
        if(dilation_w > 1)
            kernel_w += (kernel_w-1) * (dilation_w-1);

        std::vector<int> tensor_shape = {kernel_c, kernel_h, kernel_w};

        Batchf weights_f(kernels, Tensor<float>(tensor_shape));

        int non_empty_h = dilation_h > 1 ? (kernel_h+dilation_h-1)/dilation_h : kernel_h;
        int non_empty_w = dilation_w > 1 ? (kernel_w+dilation_w-1)/dilation_w : kernel_w;


        auto convert_to_float = [](std::vector<char>& attr_val, Check& check){
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

        check(kernels*kernel_c*non_empty_h*non_empty_w == weights_converted.size()) << "failed to create layer transposed convolution; kernel shape is mismatching with number of elements";

        const int kernel_hw = non_empty_w * non_empty_h;

        check(kernels % num_groups == 0) << "failed to create layer transposed convolution; number of kernels should be divisible by number of groups";
        int kernels_per_group = kernels / num_groups;

        const int kernel_nhw = kernel_hw * kernels_per_group;
        const int kernel_cnhw = kernel_nhw * kernel_c;

        for(int g=0; g<num_groups; g++) {
            std::vector<float> sub_weights = {weights_converted.begin()+g*kernel_cnhw, weights_converted.begin()+(g+1)*kernel_cnhw};
            for(int n=0; n<kernels_per_group; n++) {
                int offset_wrt_kernel = n*kernel_hw;
                for(int c=0; c<kernel_c; c++) {
                    int offset_wrt_channel = c*kernel_nhw;
                    for(int h=0; h<kernel_h; h++) {
                        int offset_wrt_height = h*kernel_c;
                        for(int w=0; w<kernel_c; w++) {
                            int h_index = (h+1)*dilation_h-1;
                            int w_index = (w+1)*dilation_w-1;
                            weights_f[kernels_per_group*g+n].WriteMatrix(c)(h_index, w_index) = sub_weights[offset_wrt_channel+offset_wrt_kernel+offset_wrt_height+w];
                        }
                    }
                }
            }
        }

        Eigen::RowVectorXf bias_f;

        if(use_bias->second.b) {
            auto bias = opt->attrs.find("bias");
            check(bias != opt->attrs.end()) << "failed to create layer transposed convolution; cannot find attribute bias";

            std::vector<float> bias_farr = convert_to_float(bias->second.data, check);

            bias_f = Eigen::Map<Eigen::RowVectorXf>(bias_farr.data(), bias_farr.size());
        }

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerTransposedConvolution>(std::move(opt->name),
                                                  std::move(opt->inputnames), std::move(output_name),
                                                  std::move(weights_f),
                                                  use_bias->second.b, bias_f,
                                                  padding->second.ai[0],padding->second.ai[1],
                                                  stride->second.ai[0], stride->second.ai[1],
                                                  groups->second.i,
                                                  output_padding->second.ai[0], output_padding->second.ai[1]);

    };
}

