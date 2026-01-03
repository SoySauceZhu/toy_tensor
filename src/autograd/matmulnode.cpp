//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#include "../../include/mytensor/autograd/matmulnode.h"

#include "mytensor/tensor.h"

namespace mytensor {
    MatMulNode::MatMulNode(std::shared_ptr<Tensor<float> > intput_a,
                     std::shared_ptr<Tensor<float> > input_b)
        : a_(std::move(intput_a)), b_(std::move(input_b)) {
        inputs.push_back(a_);
        inputs.push_back(b_);
    }

    /*
     * >>> z = y + x
     * >>> dz/dx = 1
     *
     * >>> L = loss(z)
     *
     * >>> dL/dx = dL/dz * dz/dx = dL/dz * 1 = dL/dz
     *
     */

    void MatMulNode::backward(const Tensor<float> &grad_output) {
        if (a_->requires_grad()) {
            if (!a_->grad()) {
                a_->grad() = std::make_shared<Tensor<float> >(a_->shape(), 0.0f);
            }
            // *(a_->grad()) += grad_output;
            // TODO

        }
    }
}
