#ifndef MYTENSOR_AUTOGRAD_H
#define MYTENSOR_AUTOGRAD_H

#include <iostream>
#include <memory>
#include <vector>

#include "autograd.h"

namespace mytensor {
    template<typename T>
    class Tensor;

    template<typename T>
    class AutogradNode {
    public:
        virtual ~AutogradNode() = default;

        // NO Constructor, should be defined in children classes

        // grad_output = dz/dy
        virtual void backward(const Tensor<T> &grad_output) = 0;

        virtual const char *name() = 0;

        // friend std::ostream &operator<<(std::ostream &os, const AutogradNode<T> &node) {
        //     os << node.name();
        //     return os;
        // }

        // input = x, y, ...
        std::vector<std::shared_ptr<Tensor<T> > > inputs;
    };
} // namespace mytensor

#endif // MYTENSOR_AUTOGRAD_H
