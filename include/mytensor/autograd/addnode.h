//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_ADDNODE_H
#define MYTENSOR_ADDNODE_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include "autograd.h"

namespace mytensor {
    template<typename T>
    class AddNode final : public AutogradNode<T> {
    public:
        AddNode(std::shared_ptr<Tensor<T> > intput_a,
                std::shared_ptr<Tensor<T> > input_b)
            : a_(std::move(intput_a)), b_(std::move(input_b)) {
            this->inputs.push_back(a_);
            this->inputs.push_back(b_);
        }

        const char *name() override { return "AddNode"; }


        void backward(const Tensor<T> &grad_output) override {
            if (a_ && a_->requires_grad()) {
                accumulate_grad(a_->grad(), grad_output);
            }
            if (b_ && b_->requires_grad()) {
                accumulate_grad(b_->grad(), grad_output);
            }
        }

    private:
        /*
        * dz/dx = 1
        * dL/dx = dL/dz * dz/dx = dL/dz
        */
        static void accumulate_grad(std::shared_ptr<Tensor<T> > &target, const Tensor<T> &grad_output) {
            // if input tensor has no grad_, then initialize with zero tensor
            if (!target) {
                target = std::make_shared<Tensor<T> >(grad_output.shape(), T(), false);
            }
            const size_t n = grad_output.numel();
            for (size_t i = 0; i < n; ++i) {
                (*target)[i] += grad_output[i];
            }
        }

        std::shared_ptr<Tensor<T> > a_;
        std::shared_ptr<Tensor<T> > b_;
    };
}


#endif //MYTENSOR_ADDNODE_H
