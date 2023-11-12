#include "layer/concrete/cat.hpp"


namespace sky_infer {
    LayerCat::LayerCat(std::string name, std::vector<std::string> input_name, std::vector<std::string> output_name,
                       int dim) {
        type_ = LayerType::Cat;
        name_ = std::move(name);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
        dim_ = dim;
    }


    void LayerCat::Forward() {

    }






}