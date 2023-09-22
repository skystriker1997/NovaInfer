#include "layer/concrete/maxpooling.hpp"


namespace sky_infer {


    void LayerMaxpooling::Forward(sky_infer::Operator *opt) {
        auto item_stride = opt->params_.find("stride");
        CHECK(item_stride != opt->params_.end()) << "failed to execute maxpooling for operator " << opt->name_ << "; it misses stride parameter";

        auto item_padding = opt->params_.find("padding");
        CHECK(item_padding != opt->params_.end()) << "failed to execute maxpooling for operator " << opt->name_ << "; it misses stride parameter";

        auto item_kernel_size = opt->params_.find("kernel_size");
        CHECK(item_kernel_size != opt->params_.end()) << "failed to execute maxpooling for operator " << opt->name_ << "; it misses kernel size parameter";

        const std::vector<int>& stride = dynamic_cast<ParameterIntArray*>(item_stride->second.get())->value_;

        const std::vector<int>& padding = dynamic_cast<ParameterIntArray*>(item_padding->second.get())->value_;

        const std::vector<int>& kernel_size = dynamic_cast<ParameterIntArray*>(item_kernel_size->second.get())->value_;

        CHECK(stride.size()==2) << "failed to execute maxpooling for operator " << opt->name_ << "; size of vector stride is " << stride.size();
        CHECK(padding.size()==2) << "failed to execute maxpooling for operator " << opt->name_ << "; size of vector padding is " << padding.size();
        CHECK(kernel_size.size()==2) << "failed to execute maxpooling for operator " << opt->name_ << "; size of vector kernel_size is " << kernel_size.size();

        CHECK(opt->inputs_.size()==1 && opt->outputs_.size()==1) << "failed to execute maxpooling for operator " << opt->name_ << "; maxpooling only supports one input and one output";

        Operand* input_opd = opt->inputs_[0];
        Operand* output_opd = opt->inputs_[0];

       // CHECK(input_opd->data_.size() == output_opd->data_.size())

        CHECK(input_opd->shape_[0] == output_opd->shape_[0]) << "failed to execute maxpooling for operator " << opt->name_ << "; the input operand and output operand do not have same number of batch";
        CHECK(input_opd->shape_[1] == output_opd->shape_[1]) << "failed to execute maxpooling for operator " << opt->name_ << "; the input operand and output operand do not have same number of channel";


        CHECK(stride[0]>0 && stride[1]>0) << "failed to execute maxpooling for operator " << opt->name_ << "; the stride height and stride width must be positive integer";

        CHECK(kernel_size[0]>0 && kernel_size[1]>0) << "failed to execute maxpooling for operator " << opt->name_ << "; the kernel height and kernel width must be positive integer";

        const auto& input_shape = input_opd->shape_;

        int output_height = std::floor(((input_shape[2] + 2*padding[0]) - kernel_size[0]) / stride[0] + 1);
        int output_width = std::floor(((input_shape[3] + 2*padding[1]) - kernel_size[1]) / stride[1] + 1);

        CHECK(output_opd->shape_[2] == output_height) << "failed to execute maxpooling for operator " << opt->name_ << "; incorrect row number of output";
        CHECK(output_opd->shape_[3] == output_width) << "failed to execute maxpooling for operator " << opt->name_ << "; incorrect col number of output";

        int batch = input_opd->data_.size();

        for(int i=0; i<batch; i++) {
            Tensor<float> tensor = input_opd->data_[i];
            tensor.PaddingInpalce(std::vector<int>{padding[0], padding[1], padding[0], padding[1]}, tensor.Min());
            for(int j=0; j<output_opd->shape_[1]; j++) {
                for(int k=0; k<output_height; k++) {
                    for(int t=0; t<output_width; t++) {
                        output_opd->data_[i].WriteMatrix(j)(k,t) = tensor.ReadMatrix(j).block(k*stride[0], k*stride[1], kernel_size[0], kernel_size[1]).maxCoeff();
                    }
                }
            }
        }
    }


}