#ifndef NOVA_INFER_TENSOR
#define NOVA_INFER_TENSOR


#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include "inf_util.hpp"
#include <iostream>
#include <omp.h>

namespace nova_infer {

    template<typename T>
    class Tensor {
    private:
        std::vector<int> shape_;

        std::vector<Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data_;

        Check check_;

    public:
        Tensor() = default;

        explicit Tensor(const std::vector<int> &shape);

        Tensor(const std::vector<int> &shape, T value);

        Tensor(const std::vector<int> &shape, const std::vector<T> &data);

        Tensor(const Tensor<T> &rhs);

        Tensor(Tensor<T> &&rhs) noexcept;

        Tensor<T> &operator=(const Tensor<T> &rhs);

        Tensor<T> &operator=(Tensor<T> &&rhs) noexcept;

        void Swap(Tensor<T> &rhs) noexcept;

        ~Tensor() = default;

        int Channels() const;
        int Rows() const;
        int Cols() const;

        Tensor<T> Reshape(const std::vector<int> &shape);
        void ReshapeInplace(const std::vector<int> &shape);

        Tensor<T> Padding(const std::vector<int> &pads, T padding_value);
        void PaddingInpalce(const std::vector<int> &pads, T padding_value);

        T Max();
        T Min();

        Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ReadMatrix(int n);
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> WriteMatrix(int n);


        Tensor<T> operator+(const Tensor<T> &rhs);

        Tensor<T> operator*(const Tensor<T> &rhs);

        Tensor<T> operator%(const Tensor<T> &rhs);

        void Print();

    };


    using Batchf = std::vector<Tensor<float>>;


    template<typename T>
    void Swap(Tensor<T> &lhs, Tensor<T> &rhs) noexcept {
        lhs.Swap(rhs);
    }



    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape) {

        shape_.resize(3);

        check_(!shape.empty() && shape.size() <= 3) << "failed to construct tensor; number of dimension is " + std::to_string(shape.size());

        for(int i: shape) {
            check_(i > 0) << "failed to construct tensor; any dimension cannot be less than 1";
        }

        if(shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = 1;
            shape_[2] = shape[0];
        } else if(shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(shape_[0], Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(shape_[1], shape_[2]));
        data_ = std::move(data);
    }



    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T value) {
        shape_.resize(3);
        check_(!shape.empty() && shape.size() <= 3)
        << "failed to construct tensor; number of dimension is " + std::to_string(shape.size());
        for(int i: shape) {
            check_(i > 0) << "failed to construct tensor; any dimension cannot be less than 1";
        }
        if(shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = 1;
            shape_[2] = shape[0];
        } else if(shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(shape_[0], Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(shape_[1], shape_[2], value));
        data_ = std::move(data);
    }


    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, const std::vector<T> &data) {
        shape_.resize(3);
        check_(!shape.empty() && shape.size() <= 3)
        << "failed to construct tensor; number of dimension is " + std::to_string(shape.size());

        int i = 1;
        for(int j: shape) {
            check_(j > 0) << "failed to construct tensor; any dimension cannot be less than 1";
            i *= j;
        }
        check_(i == data.size()) << "failed to construct tensor; size of tensor does not match with shape";

        if(shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = 1;
            shape_[2] = shape[0];
        } else if(shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }

        data_.resize(shape_[0]);
        for(int k = 0; k < shape_[0]; k++) {
            data_[k] = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    std::vector<T>(data.begin() + k * shape_[1] * shape_[2],
                                   data.begin() + (k + 1) * shape_[1] * shape_[2]).data(),
                                   shape_[1],
                                   shape_[2]);
        }

    }


    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &rhs) {
        data_ = rhs.data_;
        shape_ = rhs.shape_;
    }


    template<typename T>
    Tensor<T>::Tensor(Tensor<T> &&rhs) noexcept {
        shape_ = std::move(rhs.shape_);
        data_ = std::move(rhs.data_);
    }



    template<typename T>
    Tensor<T> &Tensor<T>::operator=(const Tensor<T> &rhs) {
        data_ = rhs.data_;
        shape_ = rhs.shape_;
        return *this;
    };


    template<typename T>
    Tensor<T> &Tensor<T>::operator=(Tensor<T> &&rhs) noexcept {
        shape_ = std::move(rhs.shape_);
        data_ = std::move(rhs.data_);
        return *this;
    }


    template<typename T>
    void Tensor<T>::Swap(Tensor<T> &rhs) noexcept {
        std::swap(shape_, rhs.shape_);
        std::swap(data_, rhs.data_);
    }



    template<typename T>
    Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Tensor<T>::ReadMatrix(int n) {
        check_(n >= 0 && n < shape_[0]) << "failed to access target channel; index out of range";
        return data_[n];
    };



    template<typename T>
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Tensor<T>::WriteMatrix(int n) {
        check_(n >= 0 && n < shape_[0]) << "failed to access target channel; index out of range";
        return data_[n];
    };



    template<typename T>
    int Tensor<T>::Channels() const{
        return shape_[0];
    }



    template<typename T>
    int Tensor<T>::Rows() const{
        return shape_[1];
    }



    template<typename T>
    int Tensor<T>::Cols() const{
        return shape_[2];
    }



    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &rhs) {
        bool normal = shape_[0] == rhs.shape_[0] && shape_[1] == rhs.shape_[1] && shape_[2] == rhs.shape_[2];

        bool left_broadcast_along_channel = false;
        bool left_broadcast_along_row = false;
        bool left_broadcast_along_col = false;

        bool right_broadcast_along_channel = false;
        bool right_broadcast_along_row = false;
        bool right_broadcast_along_col = false;

        if(!normal) {
            if(shape_[0] == 1 && shape_[1] == rhs.shape_[1] && shape_[2] == rhs.shape_[2])
                left_broadcast_along_channel = true;
            else if(rhs.shape_[0] == 1 && shape_[1] == rhs.shape_[1] && shape_[2] == rhs.shape_[2])
                right_broadcast_along_channel = true;
            else if(shape_[0] == rhs.shape_[0] && shape_[1] == 1 && shape_[2] == rhs.shape_[2])
                left_broadcast_along_row = true;
            else if(shape_[0] == rhs.shape_[0] && rhs.shape_[1] == 1 && shape_[2] == rhs.shape_[2])
                right_broadcast_along_row = true;
            else if(shape_[0] == rhs.shape_[0] && shape_[1] == rhs.shape_[1] && shape_[2] == 1)
                left_broadcast_along_col = true;
            else if(shape_[0] == rhs.shape_[0] && shape_[1] == rhs.shape_[1] && rhs.shape_[2] == 1)
                right_broadcast_along_col = true;
            else
                check_(false) << "failed to add tensors; mismatching of dimensions";
        }

        if(!normal) {
            if(left_broadcast_along_channel || left_broadcast_along_col || left_broadcast_along_row) {
                check_(!right_broadcast_along_row && !right_broadcast_along_col && !right_broadcast_along_channel)
                << "failed to add tensors; at most one tensor can broadcast";
            }
            if(right_broadcast_along_row || right_broadcast_along_col || right_broadcast_along_channel) {
                check_(!left_broadcast_along_channel && !left_broadcast_along_col && !left_broadcast_along_row)
                << "failed to add tensors; at most one tensor can broadcast";
            }
        }

        int max_channel = shape_[0] > rhs.shape_[0] ? shape_[0] : rhs.shape_[0];
        int max_row = shape_[1] > rhs.shape_[1] ? shape_[1] : rhs.shape_[1];
        int max_col = shape_[2] > rhs.shape_[2] ? shape_[2] : rhs.shape_[2];

        Tensor<T> output(std::vector<int>{max_channel, max_row, max_col});


        if(normal) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i] + rhs.data_[i];
        } else if(left_broadcast_along_row && left_broadcast_along_col && left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[0](0, 0) + rhs.data_[i].array();
        } else if(left_broadcast_along_row && left_broadcast_along_col && !left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i](0, 0) + rhs.data_[i].array();
        } else if(left_broadcast_along_row && !left_broadcast_along_col && left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[0]).transpose();
        } else if(!left_broadcast_along_row && left_broadcast_along_col && left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[0]);
        } else if(left_broadcast_along_row && !left_broadcast_along_col && !left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[i]).transpose();
        } else if(!left_broadcast_along_row && left_broadcast_along_col && !left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[i]);
        } else if(!left_broadcast_along_row && !left_broadcast_along_col && left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[0] + rhs.data_[i];
        } else if(right_broadcast_along_row && right_broadcast_along_col && right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[0](0, 0) + data_[i].array();
        } else if(right_broadcast_along_row && right_broadcast_along_col && !right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[i](0, 0) + data_[i].array();
        } else if(right_broadcast_along_row && !right_broadcast_along_col && right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(rhs.data_[0]).transpose();
        } else if(!right_broadcast_along_row && right_broadcast_along_col && right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(rhs.data_[0]);
        } else if(right_broadcast_along_row && !right_broadcast_along_col && !right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(rhs.data_[i]).transpose();
        } else if(!right_broadcast_along_row && right_broadcast_along_col && !right_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(rhs.data_[i]);
        } else {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = rhs.data_[0] + data_[i];
        }
        return output;
    }


    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &rhs) {
        check_(shape_[2] == rhs.shape_[1])
        << "failed to matrix-wisely multiply tensors; left matrix col num not matching with row num of right one";

        bool normal = shape_[0] == rhs.shape_[0];
        bool left_broadcast_along_channel = false;
        bool right_broadcast_along_channel = false;
        if(!normal) {
            if(shape_[0] == 1)
                left_broadcast_along_channel = true;
            else if(rhs.shape_[0] == 1)
                right_broadcast_along_channel = true;
            else
                check_(false) << "failed to matrix-wisely multiply tensors; mismatching channels";
        }

        int output_channels = shape_[0]>rhs.shape_[0]?shape_[0]:rhs.shape_[0];

        Tensor<T> output(std::vector<int>{output_channels, shape_[1], rhs.shape_[2]});

        output.data_[0].setZero();
        if(normal) {
            for(int i = 0; i < output_channels; i++)
                output.data_[i] = data_[i] * rhs.data_[i];
        } else if(left_broadcast_along_channel) {
            for(int i = 0; i < output_channels; i++)
                output.data_[i] = data_[0] * rhs.data_[i];
        } else {
            for(int i = 0; i < output_channels; i++)
                output.data_[i] = data_[i] * rhs.data_[0];
        }
        return output;
    }


    template<typename T>
    Tensor<T> Tensor<T>::operator%(const Tensor<T> &rhs) {
        check_(shape_[1] == rhs.shape_[1] && shape_[2] == rhs.shape_[2])
        << "failed to coef-wisely multiply tensors; mismatching shapes";

        bool normal = shape_[0] == rhs.shape_[0];

        bool left_broadcast_along_channel = shape_[0] != rhs.shape_[0] && shape_[0] == 1;
        bool right_broadcast_along_channel = shape_[0] != rhs.shape_[0] && rhs.shape_[0] == 1;

        check_(normal || left_broadcast_along_channel || right_broadcast_along_channel)
        << "failed to coef-wisely multiply tensors; mismatching shapes";

        Tensor<T> output(std::vector<int>{1, shape_[1], shape_[2]});

        if(normal) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[0].array() += data_[i].array() * rhs.data_[i].array();
        } else if(left_broadcast_along_channel) {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[0].array() += data_[0].array() * rhs.data_[i].array();
        } else {
            for(int i = 0; i < output.shape_[0]; i++)
                output.data_[0].array() += data_[i].array() * rhs.data_[0].array();
        }

        return output;
    }


    template<typename T>
    T Tensor<T>::Min() {
        T i = data_[0].minCoeff();
        for(int j=0; j<data_.size(); j++) {
            if(j==0)
                continue;
            int m = data_[j].minCoeff();
            i = m < i ? m : i;
        }
        return i;
    }


    template<typename T>
    T Tensor<T>::Max() {
        T i = data_[0].maxCoeff();
        for(int j=0; j<data_.size(); j++) {
            if(j==0)
                continue;
            int m = data_[j].maxCoeff();
            i = m > i ? m : i;
        }
        return i;
    }



    template<typename T>
    Tensor<T> Tensor<T>::Padding(const std::vector<int> &pads, T padding_value) {
        // pads: upper, left, bottom, right
        check_(pads.size() == 4) << "failed to padding; size of padding vector is not exactly 4";
        for(int i: pads)
            check_(i >= 0) << "failed to padding; elements of padding vector cannot be negative";
        std::vector<int> shape{shape_[0], shape_[1] + pads[0] + pads[2], shape_[2] + pads[1] + pads[3]};
        Tensor<T> tmp(shape, padding_value);
        for(int i=0; i<shape[0]; i++)
            tmp.data_[i].block(pads[0], pads[1], shape_[1], shape_[2]) = data_[i];
        return tmp;
    }


    template<typename T>
    void Tensor<T>::PaddingInpalce(const std::vector<int> &pads, T padding_value) {
        check_(pads.size() == 4) << "failed to padding; size of padding vector is not exactly 4";
        for(int i: pads)
            check_(i >= 0) << "failed to padding; elements of padding vector cannot be negative";
        std::vector<int> shape{shape_[0], shape_[1] + pads[0] + pads[2], shape_[2] + pads[1] + pads[3]};
        Tensor<T> tmp(shape, padding_value);
        for(int i=0; i<shape[0]; i++)
            tmp.data_[i].block(pads[0], pads[1], shape_[1], shape_[2]) = data_[i];
        data_ = std::move(tmp.data_);
        shape_ = std::move(shape);
    }




    template<typename T>
    void Tensor<T>::ReshapeInplace(const std::vector<int> &shape) {
        check_(!shape.empty() && shape.size()<=3) << "failed to reshape; improper size of shape argument";

        std::vector<int> shape_to;
        if(shape.size()==1)
            shape_to = {1, 1, shape[0]};
        else if(shape.size()==2)
            shape_to = {1, shape[0], shape[1]};
        else
            shape_to = shape;

        check_(shape_to[0]>0 && shape_to[1]>0 && shape_to[2]>0) << "failed to reshape; any dimension cannot be less than 1";
        check_(shape_to[0]*shape_to[1]*shape_to[2] == shape_[0]*shape_[1]*shape_[2]) << "failed to reshape; mismatching with number of elements";

        if(shape_to[0] != shape_[0]) {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flat(1,  shape_[0] * shape_[1] * shape_[2]);
            for(int i = 0; i < shape_[0]; i++) {
                flat.block(0, shape_[1] * shape_[2] * i, 1, shape_[1] * shape_[2]) = data_[i].template reshaped<Eigen::RowMajor>(1, shape_[1] * shape_[2]);
            }
            std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> tmp_data(shape_to[0]);
            for(int j = 0; j < shape_to[0]; j++) {
                tmp_data[j] = flat.block(0, shape_to[1] * shape_to[2] * j, 1, shape_to[1] * shape_to[2]).template reshaped<Eigen::RowMajor>(
                        shape_to[1], shape_to[2]);
            }
            data_ = std::move(tmp_data);
        } else {
            for(auto& mtx: data_) {
                mtx.resize(shape_to[1], shape_to[2]);
            }
        }

        shape_ = std::move(shape_to);
    }


    template<typename T>
    Tensor<T> Tensor<T>::Reshape(const std::vector<int> &shape) {
        check_(!shape.empty() && shape.size()<=3) << "failed to reshape; improper size of shape argument";

        std::vector<int> shape_to;
        if(shape.size()==1)
            shape_to = {1, 1, shape[0]};
        else if(shape.size()==2)
            shape_to = {1, shape[0], shape[1]};
        else
            shape_to = shape;

        check_(shape_to[0]>0 && shape_to[1]>0 && shape_to[2]>0) << "failed to reshape; any dimension cannot be less than 1";
        check_(shape_to[0]*shape_to[1]*shape_to[2] == shape_[0]*shape_[1]*shape_[2]) << "failed to reshape; mismatching with number of elements";

        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> tmp_data(shape_to[0]);

        if(shape_to[0] != shape_[0]) {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flat(1,  shape_[0] * shape_[1] * shape_[2]);
            for(int i = 0; i < shape_[0]; i++) {
                flat.block(0, shape_[1] * shape_[2] * i, 1, shape_[1] * shape_[2]) = data_[i].template reshaped<Eigen::RowMajor>(1, shape_[1] * shape_[2]);
            }
            for(int j = 0; j < shape_to[0]; j++) {
                tmp_data[j] = flat.block(0, shape_to[1] * shape_to[2] * j, 1, shape_to[1] * shape_to[2]).template reshaped<Eigen::RowMajor>(shape_to[1], shape_to[2]);
            }
        } else {
            for(int k = 0; k < shape_to[0]; k++) {
                tmp_data[k] = data_[k].template reshaped<Eigen::RowMajor>(shape_to[1], shape_to[2]);
            }
        }

        Tensor<T> output;
        output.data_ = std::move(tmp_data);
        output.shape_ = std::move(shape_to);

        return output;
    }




    template<typename T>
    void Tensor<T>::Print() {
        for(int i=0; i<data_.size(); i++) {
            std::cout << "channel " << i << ": " << std::endl;
            std::cout << data_[i] << std::endl;
        }
    }





}




#endif



