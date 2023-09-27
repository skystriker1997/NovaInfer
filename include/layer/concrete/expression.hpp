#ifndef SKY_INFER_LAYER_EXPRESSION
#define SKY_INFER_LAYER_EXPRESSION

#include "layer/layer.hpp"
//#include "layer/concrete/expression/parser.hpp"


namespace sky_infer {


    class LayerExpression: public Layer {
    private:
        LayerType type_;
        std::vector<std::shared_ptr<Batch<float>>> inputs_;
        std::shared_ptr<Batch<float>> output_;
        std::string expression_;
        std::vector<std::string> token_vector_;
        Check check_;

        void Parse();

        friend class Graph;

    public:
        LayerExpression(std::vector<std::shared_ptr<Batch<float>>>&& inputs,std::shared_ptr<Batch<float>> output, std::string&& expression);

        void Forward() override;

        ~LayerExpression() override = default;
    };

}




#endif