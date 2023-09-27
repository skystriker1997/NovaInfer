#include "layer/concrete/maxpooling.hpp"


namespace sky_infer {


    LayerMaxpooling::LayerMaxpooling(std::shared_ptr<Batch<float>> input,
                                     std::shared_ptr<Batch<float>> output,
                                     std::vector<int>&& stride,
                                     std::vector<int>&& padding,
                                     std::vector<int>&& kernel_size) :
                                     input_(std::move(input)), output_(std::move(output)), stride_(stride), padding_(padding), kernel_size_(kernel_size) {};

    void LayerMaxpooling::Forward() {

        check_(stride_.size()==2) << "failed to execute maxpooling; size of vector stride is " + std::to_string(stride_.size());
        check_(stride_[0]>0 && stride_[1]>0) << "failed to execute maxpooling; the stride height and stride width must be positive integer";

        check_(padding_.size()==2) << "failed to execute maxpooling; size of vector padding is " + std::to_string(padding_.size());
        check_(padding_[0]>0 && padding_[1]>0) << "failed to execute maxpooling; the padding height and padding width must be positive integer";

        check_(kernel_size_.size()==2) << "failed to execute maxpooling for operator; size of vector kernel_size is " + std::to_string(kernel_size_.size());
        check_(kernel_size_[0]>0 && kernel_size_[1]>0) << "failed to execute; the kernel height and kernel width must be positive integer";


        check_(input_->shape_[0] == output_->shape_[0]) << "failed to execute maxpooling; the input and output do not have same number of batch";
        check_(input_->shape_[1] == output_->shape_[1]) << "failed to execute maxpooling; the input and output do not have same number of channel";


        int output_height = std::floor(((input_->shape_[2] + 2*padding_[0]) - kernel_size_[0]) / stride_[0] + 1);
        int output_width = std::floor(((input_->shape_[3] + 2*padding_[1]) - kernel_size_[1]) / stride_[1] + 1);

        check_(output_->shape_[2] == output_height) << "failed to execute maxpooling; incorrect row number of output";
        check_(output_->shape_[3] == output_width) << "failed to execute maxpooling; incorrect col number of output";

        int batch_size = input_->shape_[0];

        for(int i=0; i<batch_size; i++) {
            Tensor<float> tensor = input_->data_[i];
            tensor.PaddingInpalce(std::vector<int>{padding_[0], padding_[1], padding_[0], padding_[1]}, tensor.Min());
            for(int j=0; j<output_->data_[0].ReadShape()[0]; j++) {
                for(int k=0; k<output_height; k++) {
                    for(int t=0; t<output_width; t++) {
                        output_->data_[i].WriteMatrix(j)(k,t) = tensor.ReadMatrix(j).block(k*stride_[0], k*stride_[1], kernel_size_[0], kernel_size_[1]).maxCoeff();
                    }
                }
            }
        }
    }


}