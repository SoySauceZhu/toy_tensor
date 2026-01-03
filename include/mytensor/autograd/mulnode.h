//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_MULNODE_H
#define MYTENSOR_MULNODE_H
#include "autograd.h"


namespace mytensor {
    template<typename T>
    class MulNode final : public AutogradNode<T> {
    public:
        MulNode(Tensor<T> *intput_a,
                Tensor<T> *input_b)
            : a_(intput_a), b_(input_b) {
            this->inputs.push_back(a_);
            this->inputs.push_back(b_);
        }

        void backward(const Tensor<T> &grad_output) override {
            if (a_ && a_->requires_grad()) {
                accumulate_grad(a_->grad(), grad_output, b_);
            }
            if (b_ && b_->requires_grad()) {
                accumulate_grad(b_->grad(), grad_output, a_);
            }
        }

        const char *name() override { return "MulNode"; };

    private:
        Tensor<T> *a_;
        Tensor<T> *b_;


        /*
         * dz/dx = y
         * dL/dx = dL/dz * dz/dx = dL/dz * y
         */

        static void accumulate_grad(std::shared_ptr<Tensor<T> > &grad_input, const Tensor<T> &grad_output,
                                    const Tensor<T> *other_tensor_input) {
            if (!grad_input) {
                grad_input = std::make_shared<Tensor<T> >(grad_output.shape(), T(), false);
            }

            const size_t n = grad_output.numel();
            for (size_t i = 0; i < n; i++) {
                (*grad_input)[i] += (*other_tensor_input)[i] * grad_output[i];
            }
        }
    };
}


#endif //MYTENSOR_MULNODE_H
