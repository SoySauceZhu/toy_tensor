#ifndef MYTENSOR_OPS_H
#define MYTENSOR_OPS_H

#include <memory>
#include <stdexcept>

#include "mytensor/autograd/addnode.h"
#include "mytensor/tensor.h"

namespace mytensor {
    // Core op APIs (shared_ptr-based graph-friendly style)
    template<typename T>
    std::shared_ptr<Tensor<T> > add(const std::shared_ptr<Tensor<T> > &a,
                                    const std::shared_ptr<Tensor<T> > &b) {
        if (!a || !b) {
            throw std::invalid_argument("add: null tensor");
        }
        if (a->shape() != b->shape()) {
            throw std::invalid_argument("add: shape mismatch");
        }

        auto c = std::make_shared<Tensor<T> >(a->shape(), T(),
                                              a->requires_grad() || b->requires_grad());
        const size_t num = a->numel();
        for (size_t i = 0; i < num; ++i) {
            (*c)[i] = (*a)[i] + (*b)[i];
        }

        if (c->requires_grad()) {
            c->set_grad_fn(std::make_shared<AddNode<T> >(a, b));
        }

        return c;
    }

    template<typename T>
    std::shared_ptr<Tensor<T> > operator+(const std::shared_ptr<Tensor<T> > &a,
                                          const std::shared_ptr<Tensor<T> > &b) {
        return add(a, b);
    }


} // namespace mytensor::ops

#endif // MYTENSOR_OPS_H
