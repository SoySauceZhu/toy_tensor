#ifndef MYTENSOR_OPS_H
#define MYTENSOR_OPS_H

#include <memory>
#include <stdexcept>

#include "kernals.h"
#include "autograd/addnode.h"


namespace mytensor {
    // Core op APIs (shared_ptr-based graph-friendly style)
    template<typename T>
    Tensor<T> operator+(const Tensor<T> &a,
                        const Tensor<T> &b) {
        if (!a || !b) {
            throw std::invalid_argument("mytensor::ops::add_kernel: null input");
        }
        if (a.numel() != b.numel()) {
            throw std::invalid_argument("mytensor::ops::add_kernel: size mismatch");
        }

        Tensor<T> c(a->shape(), T(),
                    a->requires_grad() || b->requires_grad());

        ops::add_kernel(*a, *b, c);


        if (a.requires_grad() || b.requires_grad()) {
            c.set_requires_grad(true);
            c.set_grad_fn(std::make_shared<AddNode>(a.shared_from_this(),
                                                    b.shared_from_this()));
        }
        return c;
    }
} // namespace mytensor

#endif // MYTENSOR_OPS_H
