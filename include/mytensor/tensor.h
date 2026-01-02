#ifndef MYTENSOR_TENSOR_H
#define MYTENSOR_TENSOR_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mytensor {
    template<typename T = float>
    class Tensor {
    public:
        Tensor(std::vector<size_t> shape, T init_val = T())
            : data_(numel_from_shape(shape), init_val), shape_(std::move(shape)) {
            compute_strides();
        }

        Tensor(std::vector<size_t> shape, std::vector<T> data)
            : data_(std::move(data)), shape_(std::move(shape)) {
            const size_t expected = numel_from_shape(shape_);
            if (data_.size() != expected) {
                throw std::invalid_argument("Tensor data size does not match shape.");
            }
            compute_strides();
        }

        // Define `[]` operator, return a reference of T element
        T &operator[](size_t idx) { return data_.at(idx); }
        const T &operator[](size_t idx) const { return data_.at(idx); }

        // Getter function, retrieve an element by {idx_row, idx_col, idx_3, ...}
        T &at(const std::vector<size_t> &indices) {
            return data_.at(offset(indices));
        }
        const T &at(const std::vector<size_t> &indices) const {
            return data_.at(offset(indices));
        }

        const std::vector<size_t> &shape() const { return shape_; }

        // number of elements
        size_t numel() const { return data_.size(); }

        void set_requires_grad(bool flag) { requires_grad_ = flag; }
        bool requires_grad() const { return requires_grad_; }

        void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

        friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
            os << "Tensor(shape=[";
            for (size_t i = 0; i < t.shape_.size(); ++i) {
                os << t.shape_[i];
                if (i + 1 < t.shape_.size()) {
                    os << ", ";
                }
            }
            os << "], requires_grad=" << (t.requires_grad_ ? "True" : "False") << "):\n";

            if (t.shape_.size() == 1) {
                os << "[";
                for (size_t i = 0; i < t.shape_[0]; ++i) {
                    os << t.data_[i];
                    if (i + 1 < t.shape_[0]) {
                        os << ", ";
                    }
                }
                os << "]";
            } else if (t.shape_.size() == 2) {
                const size_t rows = t.shape_[0];
                const size_t cols = t.shape_[1];
                os << "[";
                for (size_t r = 0; r < rows; ++r) {
                    os << "[";
                    for (size_t c = 0; c < cols; ++c) {
                        os << t.data_[r * t.strides_[0] + c * t.strides_[1]];
                        if (c + 1 < cols) {
                            os << ", ";
                        }
                    }
                    os << "]";
                    if (r + 1 < rows) {
                        os << ",\n ";
                    }
                }
                os << "]";
            } else {
                os << "[";
                for (size_t i = 0; i < t.data_.size(); ++i) {
                    os << t.data_[i];
                    if (i + 1 < t.data_.size()) {
                        os << ", ";
                    }
                }
                os << "]";
            }

            return os;
        }

    private:
        std::vector<T> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        bool requires_grad_ = false;


        /*
         * >>> shape_ = {28,28,3}
         * >>> strides_ = {84, 3, 1} // 84 = 3 * 28
         *
         */

        void compute_strides() {
            strides_.resize(shape_.size());
            if (shape_.empty()) {
                return;
            }
            strides_.back() = 1;
            for (size_t i = shape_.size() - 1; i > 0; --i) {
                strides_[i - 1] = strides_[i] * shape_[i];
            }
        }

        size_t offset(const std::vector<size_t> &indices) const {
            if (indices.size() != shape_.size()) {
                throw std::invalid_argument("Tensor indices rank mismatch.");
            }
            size_t off = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range("Tensor index out of range.");
                }
                off += indices[i] * strides_[i];
            }
            return off;
        }

        static size_t numel_from_shape(const std::vector<size_t> &shape) {
            if (shape.empty()) {
                return 0;
            }
            return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                                   std::multiplies<size_t>());
        }
    };
} // namespace mytensor

#endif // MYTENSOR_TENSOR_H
