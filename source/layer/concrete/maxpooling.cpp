#include "layer/concrete/maxpooling.hpp"


namespace nova_infer {


    LayerMaxpooling::LayerMaxpooling(std::string name,
                                     std::vector<std::string> input_name, std::vector<std::string> output_name,
                                     int stride_h, int stride_w,
                                     int padding_h, int padding_w,
                                     int kernel_h, int kernel_w)
    {
    check_(padding_h>=0 && padding_w>=0) << "failed to construct layer maxpooling; padding should be non-negative integers";
    check_(stride_h>=1 && stride_w>=1) << "failed to construct layer maxpooling; stride should be positive integers";

    check_(kernel_h>=1 && kernel_w>=1) << "failed to construct layer maxpooling; kernel should be positive integers";

    type_ = LayerType::MaxPooling;
    name_ = std::move(name);
    input_name_ = std::move(input_name);
    output_name_ = std::move(output_name);
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    padding_h_ = padding_h;
    padding_w_ = padding_w;
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    };


    void LayerMaxpooling::Forward() {

        auto batch_size = input_->size();

        omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
        for(int i=0; i<batch_size; i++) {
            Tensor<float> &in = input_->at(i);
            Tensor<float> &out = output_->at(i);
            check_(in.Channels() == out.Channels()) << "failed to execute maxpooling; the input tensor and output tensor do not have same batch size";

            int output_height = std::floor(((in.Rows() + 2*padding_h_) - kernel_h_) / stride_h_ + 1);
            int output_width = std::floor(((in.Cols() + 2*padding_w_) - kernel_w_) / stride_w_ + 1);

            check_(out.Rows() == output_height) << "failed to execute maxpooling; incorrect row number of output";
            check_(out.Cols() == output_width) << "failed to execute maxpooling; incorrect col number of output";

            Tensor<float> tensor = in;

            tensor.PaddingInpalce(std::vector<int>{padding_h_, padding_w_, padding_h_, padding_w_}, tensor.Min());

            for(int j=0; j<out.Channels(); j++) {
                for(int k=0; k<output_height; k++) {
                    for(int t=0; t<output_width; t++) {
                        out.WriteMatrix(j)(k,t) = tensor.ReadMatrix(j).block(k*stride_h_, k*stride_w_, kernel_h_, kernel_w_).maxCoeff();
                    }
                }
            }
        }
    }

    std::shared_ptr<LayerMaxpooling> MakeLayerMaxpooling(pnnx::Operator *opt) {
        Check check;
        check(opt->inputnames.size()==1) << "failed to create layer maxpooling; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer maxpooling; only produce one tensor as output";


        auto stride = opt->params.find("stride");
        check(stride != opt->params.end()) << "failed to create layer maxpooling; cannot find parameter stride";
        check(stride->second.ai.size()==2) << "failed to create layer maxpooling; the parameter stride should have 2 elements";
        auto padding = opt->params.find("padding");
        check(padding != opt->params.end()) << "failed to create layer maxpooling; cannot find parameter padding";
        check(padding->second.ai.size()==2) << "failed to create layer maxpooling; the parameter padding should have 2 elements";
        auto kernel_size = opt->params.find("kernel_size");
        check(kernel_size != opt->params.end()) << "failed to create layer maxpooling; cannot find parameter kernel_size";
        check(kernel_size->second.ai.size()==2) << "failed to create layer maxpooling; the parameter kernel_size should have 2 elements";

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerMaxpooling>(std::move(opt->name),
                                                 std::move(opt->inputnames), std::move(output_name),
                                                 stride->second.ai[0], stride->second.ai[1],
                                                 padding->second.ai[0], padding->second.ai[1],
                                                 kernel_size->second.ai[0], kernel_size->second.ai[1]);
    };

}