#ifndef SKY_INFER_LAYER_MAXPOOLING
#define SKY_INFER_LAYER_MAXPOOLING


#include "layer/layer.hpp"

namespace sky_infer {

    class LayerMaxpooling: public Layer {

    private:
        LayerType type_;
        std::shared_ptr<Batch<float>> input_;
        std::shared_ptr<Batch<float>> output_;
        std::vector<int> stride_;
        std::vector<int> padding_;
        std::vector<int> kernel_size_;
        Check check_;

        friend class Graph;

    public:
        LayerMaxpooling(std::shared_ptr<Batch<float>> input,
                        std::shared_ptr<Batch<float>> output,
                        std::vector<int>&& stride,
                        std::vector<int>&& padding,
                        std::vector<int>&& kernel_size);
        void Forward() override;

        ~LayerMaxpooling() override = default;

    };
}





#endif