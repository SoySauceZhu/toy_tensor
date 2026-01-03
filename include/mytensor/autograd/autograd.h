#ifndef MYTENSOR_AUTOGRAD_H
#define MYTENSOR_AUTOGRAD_H

#include <iostream>
#include <memory>
#include <vector>

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

        // input = x
        std::vector<std::shared_ptr<Tensor<T>>> inputs;
    };
} // namespace mytensor

#endif // MYTENSOR_AUTOGRAD_H
