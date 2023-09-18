#ifndef SKY_INFER_TENSOR
#define SKY_INFER_TENSOR


#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <glog/logging.h>
#include <iostream>




namespace sky_infer {

    template<typename T>
    class Tensor {
    private:
        std::vector<int> shape_;
        std::vector<Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic>> data_;

    public:
        Tensor() = default;

        explicit Tensor(const std::vector<int> &shape);

        explicit Tensor(const std::vector<int> &shape, T value);

        Tensor(const std::vector<int> &shape, std::vector<T> &data, bool column_major);

        Tensor(const Tensor<T> &other);

        Tensor(Tensor<T> &&other) noexcept;

        explicit Tensor(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data);
        explicit Tensor(std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&& data) noexcept;

        Tensor<T> &operator=(const Tensor<T> &other);

        Tensor<T> &operator=(Tensor<T> &&other) noexcept;

        Tensor<T> &operator=(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data);
        Tensor<T> &operator=(std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&& data) noexcept;


        ~Tensor() = default;

        Eigen::Ref<const Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic>> ReadMatrix(int n);
        Eigen::Ref<Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic>> WriteMatrix(int n);

        const std::vector<int> &ReadShape();
        std::vector<int> &WriteShape();

        Tensor<T> operator+(const Tensor<T> &other);

        Tensor<T> operator*(const Tensor<T> &other);

        Tensor<T> operator%(const Tensor<T> &other);

        void Padding(const std::vector<int>& pads, T padding_value);

        void Print();

    };

    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape) {
        shape_.resize(3);
        CHECK(!shape.empty() && shape.size() <= 3)
                        << "failed to construct tensor; number of dimension is " << shape.size();
        for (int i: shape) {
            CHECK(i > 0) << "failed to construct tensor; any dimension cannot be less than 1";
            //  i *= j;
        }
        if (shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = 1;
        } else if (shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> data(shape_[0],
                                                                           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(
                                                                                   shape_[1], shape_[2]));
        data_ = std::move(data);
    }



    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, T value) {
        shape_.resize(3);
        CHECK(!shape.empty() && shape.size() <= 3)
                        << "failed to construct tensor; number of dimension is " << shape.size();
        for (int i: shape) {
            CHECK(i > 0) << "failed to construct tensor; any dimension cannot be less than 1";
            //  i *= j;
        }
        if (shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = 1;
        } else if (shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> data(shape_[0],
                                                                           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(
                                                                                   shape_[1], shape_[2], value));
        data_ = std::move(data);
    }


    template<typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape, std::vector<T> &data, bool column_major) {
        shape_.resize(3);
        CHECK(!shape.empty() && shape.size() <= 3)
                        << "failed to construct tensor; number of dimension is " << shape.size();

        int i = 1;
        for (int j: shape) {
            CHECK(j > 0) << "failed to construct tensor; any dimension cannot be less than 1";
            i *= j;
        }
        CHECK(i == data.size()) << "failed to construct tensor; size of data does not match with shape";

        if (shape.size() == 1) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = 1;
        } else if (shape.size() == 2) {
            shape_[0] = 1;
            shape_[1] = shape[0];
            shape_[2] = shape[1];
        } else {
            shape_ = shape;
        }

        //  shape_ = shape;

        data_.resize(shape_[0]);
        for (int k = 0; k < shape_[0]; k++) {
            data_[k] = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
                    std::vector<T>(data.begin() + k * shape_[1] * shape_[2],
                                   data.begin() + (k + 1) * shape_[1] * shape_[2]).data(), shape_[1], shape_[2]);
            if (!column_major)
                data_[k].transposeInPlace();
        }
    }


    template<typename T>
    Tensor<T>::Tensor(const Tensor<T> &other) {
        data_ = other.data_;
        shape_ = other.shape_;
    }


    template<typename T>
    Tensor<T>::Tensor(Tensor<T> &&other) noexcept {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }



    template<typename T>
    Tensor<T>::Tensor(const std::vector<Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic>> &data) {
        data_ = data;
        shape_ = std::vector<int>{data.size(), data[0].row(), data[0].col()};
    }


    template<typename T>
    Tensor<T>::Tensor(std::vector<Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic>> &&data) noexcept {
        data_ = data;
        shape_ = std::vector<int>{data_.size(), data_[0].row(), data_[0].col()};
    }



    template<typename T>
    Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        return *this;
    };


    template<typename T>
    Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) noexcept {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
        return *this;
    }


    template<typename T>
    Tensor<T> &Tensor<T>::operator=(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& data) {
        data_ = data;
        shape_ = std::vector<int>{data.size(), data[0].row(), data[0].col()};
        return *this;
    };


    template<typename T>
    Tensor<T> &Tensor<T>::operator=(std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&& data) noexcept{
        data_ = data;
        shape_ = std::vector<int>{data_.size(), data_[0].row(), data_[0].col()};
        return *this;
    };



    template<typename T>
    Eigen::Ref<const Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic>> Tensor<T>::ReadMatrix(int n) {
        CHECK(n >= 0 && n < shape_[0]) << "failed to access target channel; index out of range";
        return data_[n];
    };



    template<typename T>
    Eigen::Ref<Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic>> Tensor<T>::WriteMatrix(int n) {
        CHECK(n >= 0 && n < shape_[0]) << "failed to access target channel; index out of range";
        return data_[n];
    };



    template<typename T>
    const std::vector<int> &Tensor<T>::ReadShape() {
        return shape_;
    }


    template<typename T>
    std::vector<int> &Tensor<T>::WriteShape() {
        return shape_;
    }


    ////////////

    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) {
        bool normal = shape_[0] == other.shape_[0] && shape_[1] == other.shape_[1] && shape_[2] == other.shape_[2];

        bool left_broadcast_along_channel = false;
        bool left_broadcast_along_row = false;
        bool left_broadcast_along_col = false;

        bool right_broadcast_along_channel = false;
        bool right_broadcast_along_row = false;
        bool right_broadcast_along_col = false;

        if (!normal) {
            if (shape_[0] == 1 && shape_[1] == other.shape_[1] && shape_[2] == other.shape_[2])
                left_broadcast_along_channel = true;
            else if (other.shape_[0] == 1 && shape_[1] == other.shape_[1] && shape_[2] == other.shape_[2])
                right_broadcast_along_channel = true;
            else if (shape_[0] == other.shape_[0] && shape_[1] == 1 && shape_[2] == other.shape_[2])
                left_broadcast_along_row = true;
            else if (shape_[0] == other.shape_[0] && other.shape_[1] == 1 && shape_[2] == other.shape_[2])
                right_broadcast_along_row = true;
            else if (shape_[0] == other.shape_[0] && shape_[1] == other.shape_[1] && shape_[2] == 1)
                left_broadcast_along_col = true;
            else if (shape_[0] == other.shape_[0] && shape_[1] == other.shape_[1] && other.shape_[2] == 1)
                right_broadcast_along_col = true;
            else
                LOG(FATAL) << "failed to add tensors; mismatching of dimensions";
        }


        if (!normal) {
            if (left_broadcast_along_channel || left_broadcast_along_col || left_broadcast_along_row)
                CHECK(!right_broadcast_along_row && !right_broadcast_along_col && !right_broadcast_along_channel)
                                << "failed to add tensors; at most one tensor can broadcast";
            if (right_broadcast_along_row || right_broadcast_along_col || right_broadcast_along_channel)
                CHECK(!left_broadcast_along_channel && !left_broadcast_along_col && !left_broadcast_along_row)
                                << "failed to add tensors; at most one tensor can broadcast";
        }

        int max_channel = shape_[0] > other.shape_[0] ? shape_[0] : other.shape_[0];
        int max_row = shape_[1] > other.shape_[1] ? shape_[1] : other.shape_[1];
        int max_col = shape_[2] > other.shape_[2] ? shape_[2] : other.shape_[2];

        Tensor<T> output(std::vector<int>{max_channel, max_row, max_col});


        if (normal) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i] + other.data_[i];
        } else if (left_broadcast_along_row && left_broadcast_along_col && left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[0](0, 0) + other.data_[i].array();
        } else if (left_broadcast_along_row && left_broadcast_along_col && !left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i](0, 0) + other.data_[i].array();
        } else if (left_broadcast_along_row && !left_broadcast_along_col && left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[0]).transpose();
        } else if (!left_broadcast_along_row && left_broadcast_along_col && left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[0]);
        } else if (left_broadcast_along_row && !left_broadcast_along_col && !left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[i]).transpose();
        } else if (!left_broadcast_along_row && left_broadcast_along_col && !left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(data_[i]);
        } else if (!left_broadcast_along_row && !left_broadcast_along_col && left_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[0] + other.data_[i];
        } else if (right_broadcast_along_row && right_broadcast_along_col && right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[0](0, 0) + data_[i].array();
        } else if (right_broadcast_along_row && right_broadcast_along_col && !right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[i](0, 0) + data_[i].array();
        } else if (right_broadcast_along_row && !right_broadcast_along_col && right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(other.data_[0]).transpose();
        } else if (!right_broadcast_along_row && right_broadcast_along_col && right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(other.data_[0]);
        } else if (right_broadcast_along_row && !right_broadcast_along_col && !right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].rowwise() + Eigen::Vector<T, Eigen::Dynamic>(other.data_[i]).transpose();
        } else if (!right_broadcast_along_row && right_broadcast_along_col && !right_broadcast_along_channel) {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].colwise() + Eigen::Vector<T, Eigen::Dynamic>(other.data_[i]);
        } else {
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = other.data_[0] + data_[i];
        }
        return output;
    }


    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) {
        CHECK(shape_[2] == other.shape_[1])
                        << "failed to matrix-wisely multiply tensors; left matrix col num not matching with row num of right one";

        bool normal = shape_[0] == other.shape_[0];
        bool left_broadcast_along_channel = false;
        bool right_broadcast_along_channel = false;
        if (!normal) {
            if (shape_[0] == 1)
                left_broadcast_along_channel = true;
            else if (other.shape_[0] == 1)
                right_broadcast_along_channel = true;
            else
                LOG(FATAL) << "failed to matrix-wisely multiply tensors; mismatching channels";
        }

        Tensor<T> output(std::vector<int>{1, shape_[1], other.shape_[2]});

        output.data_[0].setZero();
        if (normal) {
            for (int i = 0; i < shape_[0]; i++)
                output.data_[0] += data_[i] * other.data_[i];
        } else if (left_broadcast_along_channel) {
            for (int i = 0; i < shape_[0]; i++)
                output.data_[0] += data_[0] * other.data_[i];
        } else {
            for (int i = 0; i < shape_[0]; i++)
                output.data_[0] += data_[i] * other.data_[0];
        }
        return output;
    }


    template<typename T>
    Tensor<T> Tensor<T>::operator%(const Tensor<T> &other) {
        CHECK(shape_[1] == other.shape_[1] && shape_[2] == other.shape_[2])
                        << "failed to coef-wisely multiply tensors; mismatching shapes";

        bool normal = shape_[0] == other.shape_[0];

        bool left_broadcast_along_channel = shape_[0] != other.shape_[0] && shape_[0] == 1;
        bool right_broadcast_along_channel = shape_[0] != other.shape_[0] && other.shape_[0] == 1;

        CHECK(normal || left_broadcast_along_channel || right_broadcast_along_channel)
                        << "failed to coef-wisely multiply tensors; mismatching shapes";

        int max_channel = shape_[0] > other.shape_[0] ? shape_[0] : other.shape_[0];

        Tensor<T> output(std::vector<int>{max_channel, shape_[1], shape_[2]});

        //    input1.shape_[0] == input2.shape_[0] &&

        if (normal)
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].array() * other.data_[i].array();
        else if (left_broadcast_along_channel)
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[0].array() * other.data_[i].array();
        else
            for (int i = 0; i < output.shape_[0]; i++)
                output.data_[i] = data_[i].array() * other.data_[0].array();

        return output;
    }



    template<typename T>
    void Tensor<T>::Padding(const std::vector<int> &pads, T padding_value) {
        // up, bottom, left, right
       // CHECK(!block_mode_) << "failed to padding; at block mode";
        CHECK(pads.size() == 4) << "failed to padding; size of padding vector is not exactly 4";
        for(int i: pads)
            CHECK(i >= 0) << "failed to padding; elements of padding vector cannot be negative";
        std::vector<int> shape{shape_[0], shape_[1] + pads[0] + pads[1], shape_[2] + pads[2] + pads[3]};
        Tensor<T> tmp(shape, padding_value);
        for(int i=0; i<shape[0]; i++)
            tmp.data_[i].block(pads[1], pads[2], shape_[1], shape_[2]) = data_[i];
        data_ = std::move(tmp.data_);
        shape_ = std::move(shape);
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



