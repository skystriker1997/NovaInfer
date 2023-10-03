#ifndef SKY_INFER_LAYER_EXPRESSION
#define SKY_INFER_LAYER_EXPRESSION

#include "layer/layer.hpp"
//#include "layer/concrete/expression/parser.hpp"


namespace sky_infer {


    class LayerExpression: public Layer {
    private:
        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::vector<std::shared_ptr<Batchf>> inputs_;
        std::shared_ptr<Batchf> output_;

        std::string expression_;
        std::vector<std::string> token_vector_;
//        Check check_;

        void Parse();


    public:
        LayerExpression(std::string name, std::vector<std::shared_ptr<Batchf>> inputs,  std::shared_ptr<Batchf> output, std::string expression);

        void Forward() override;

        ~LayerExpression() override = default;
    };

}




#endif