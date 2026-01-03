// include/mytensor/ops.h
#include <memory>
#include <stdexcept>
#include <utility>

#include "kernals.h"
#include "autograd/addnode.h"  // AddNode<T>
#include "tensor.h"            // Tensor<T>

namespace mytensor {

    template <typename T>
    inline Tensor<T> add(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (!a || !b) {
            throw std::invalid_argument("mytensor::add: null tensor pointer");
        }

        if (a->shape() != b->shape()) {
            throw std::invalid_argument("mytensor::add: shape mismatch");
        }

        Tensor<T> out(a->shape(), T{}, a->requires_grad() || b->requires_grad());

        ops::add_kernel(*a, *b, out);

        if (out.requires_grad()) {
            out.set_grad_fn(std::make_shared<AddNode<T>>(a, b));
        }

        return out;
    }

    template <typename T>
    inline Tensor<T> operator+(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        return add(a, b);
    }

} // namespace mytensor
