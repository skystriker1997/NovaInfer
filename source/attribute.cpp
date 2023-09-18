#include "attribute.hpp"


namespace sky_infer {


    Attribute::Attribute(const pnnx::Attribute &pnnx_attr) {
        switch (pnnx_attr.type) {
            case 1: {
                type_ = AttrDataType::AttrFloat;
                break;
            }
            default: {
                LOG(FATAL) << "failed to construct attribute; unknown attribute type: " << pnnx_attr.type;
            }
        }
        CHECK(!pnnx_attr.shape.empty() && pnnx_attr.shape.size()<5) << "failed to construct attribute; the shape has " << pnnx_attr.shape.size() << " dimensions";
        auto float_size = sizeof(float);
        CHECK(pnnx_attr.data.size() % float_size == 0) << "failed to construct attribute; data cannot be interpreted as float";

        int n=1;
        for(int k: pnnx_attr.shape)
            n *= k;
        CHECK(n == pnnx_attr.data.size() / float_size) << "failed to construct attribute; shape mismatches number of elements";

        shape_ = pnnx_attr.shape;

        std::vector<float> vect_float;

        for(auto i=0; i<pnnx_attr.data.size()/float_size; i++) {
            float f = *((float*)pnnx_attr.data.data()+i);
            vect_float.push_back(f);
        }

        if(pnnx_attr.shape.size() < 4) {
            data_.emplace_back(pnnx_attr.shape, vect_float, true);
          //  data_ = std::move(data);
        } else {
            int c = shape_[1]*shape_[2]*shape_[3];
            for(int i=0; i<shape_[0]; i++) {
                auto tmp = std::vector(vect_float.begin()+c*i, vect_float.begin()+c*(i+1));
                data_.emplace_back(std::vector(shape_.begin()+1, shape_.end()), tmp, true);
            }
        }
    }

//    Attribute::~Attribute() {
//        delete data_;
//    }


}