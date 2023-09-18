#include "operand.hpp"



namespace sky_infer {
    Operand::Operand(pnnx::Operand *pnnx_opd) {
        switch (pnnx_opd->type) {
            case 1: {
                data_type_ = OpdDataType::OpdFloat;
                break;
            }
            default: {
                LOG(FATAL) << "unsupported data type of operand: " << pnnx_opd->type << "; up to now only float is supported";
            }
        }
        auto n = pnnx_opd->shape.size();
        CHECK(n>0 && n<5) << "failed to initialise operand; improper dimension number: " << n;
        for(int i: pnnx_opd->shape)
            CHECK(i>0) << "failed to initialise operand; any dimension cannot be less than 1";

        if(n==4) {
           // data_.resize(shape_[0]);
            auto data = std::vector<Tensor<float>>(shape_[0], Tensor<float>{std::vector<int>{shape_[1], shape_[2], shape_[3]}});
            data_ = std::move(data);
            shape_ = pnnx_opd->shape;
        } else {
            data_.emplace_back(shape_);
            if(n==3)
                shape_ = {1, pnnx_opd->shape[0], pnnx_opd->shape[1], pnnx_opd->shape[2]};
            else if(n==2)
                shape_ = {1, 1, pnnx_opd->shape[0], pnnx_opd->shape[1]};
            else
                shape_ = {1, 1, pnnx_opd->shape[0], 1};
        }
//        if(n==1 || n==2)
//            data_ = new Matrix<float>(shape_);
//        else if(n==3)
//            data_ = new Cube<float>(shape_);
//        else
//            data_ = new Tesseract<float>(shape_);
        name_ = pnnx_opd->name;
    }


//    Operand::~Operand() {
//        delete data_;
//    }
}