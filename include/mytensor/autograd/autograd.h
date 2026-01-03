#ifndef MYTENSOR_AUTOGRAD_H
#define MYTENSOR_AUTOGRAD_H

#include <iostream>
#include <memory>
#include <vector>

namespace mytensor {
    template<typename T>
    class Tensor;

    class AutogradNode {
    public:
        virtual ~AutogradNode() = default;

        // NO Constructor, should be defined in children classes

        /*
         * >>> z = y**2
         * >>> y = sigmoid(x) + 2
         *
         */

        // grad_output = dz/dy
        virtual void backward(const Tensor<float> &grad_output) = 0;

        // input = x
        std::vector<std::shared_ptr<Tensor<float> > > inputs;
    };
} // namespace mytensor

#endif // MYTENSOR_AUTOGRAD_H
