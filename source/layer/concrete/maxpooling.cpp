#include "layer/concrete/maxpooling.hpp"


namespace sky_infer {


    LayerMaxpooling::LayerMaxpooling(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output, std::vector<int> stride, std::vector<int> padding, std::vector<int> kernel_size) :
                                     type_(LayerType::MaxPooling),
                                     name_(std::move(name)),
                                     input_(std::move(input)),
                                     output_(std::move(output)),
                                     stride_(std::move(stride)),
                                     padding_(std::move(padding)),
                                     kernel_size_(std::move(kernel_size)) {};

    void LayerMaxpooling::Forward() {

        check_(stride_.size()==2) << "failed to execute maxpooling; size of vector stride is " + std::to_string(stride_.size());
        check_(stride_[0]>0 && stride_[1]>0) << "failed to execute maxpooling; the stride height and stride width must be positive integer";

        check_(padding_.size()==2) << "failed to execute maxpooling; size of vector padding is " + std::to_string(padding_.size());
        check_(padding_[0]>0 && padding_[1]>0) << "failed to execute maxpooling; the padding height and padding width must be positive integer";

        check_(kernel_size_.size()==2) << "failed to execute maxpooling for operator; size of vector kernel_size is " + std::to_string(kernel_size_.size());
        check_(kernel_size_[0]>0 && kernel_size_[1]>0) << "failed to execute; the kernel height and kernel width must be positive integer";


        check_(input_->size() == output_->size()) << "failed to execute maxpooling; the input and output have different batch sizes";

        auto batch_size = input_->size();

        for(int i=0; i<batch_size; i++) {
            Tensor<float>& in = input_->at(i);
            Tensor<float>& out = output_->at(i);
            check_(in.Channels() == out.Channels()) << "failed to execute maxpooling; the input tensor and output tensor do not have same batch size";

            int output_height = std::floor(((in.Rows() + 2*padding_[0]) - kernel_size_[0]) / stride_[0] + 1);
            int output_width = std::floor(((in.Cols() + 2*padding_[1]) - kernel_size_[1]) / stride_[1] + 1);

            check_(out.Rows() == output_height) << "failed to execute maxpooling; incorrect row number of output";
            check_(out.Cols() == output_width) << "failed to execute maxpooling; incorrect col number of output";

            Tensor<float> tensor = in;

            tensor.PaddingInpalce(std::vector<int>{padding_[0], padding_[1], padding_[0], padding_[1]}, tensor.Min());

            for(int j=0; j<out.Channels(); j++) {
                for(int k=0; k<output_height; k++) {
                    for(int t=0; t<output_width; t++) {
                        out.WriteMatrix(j)(k,t) = tensor.ReadMatrix(j).block(k*stride_[0], k*stride_[1], kernel_size_[0], kernel_size_[1]).maxCoeff();
                    }
                }
            }
        }
    }


}