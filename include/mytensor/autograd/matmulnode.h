//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_MATMULNODE_H
#define MYTENSOR_MATMULNODE_H
#include "autograd.h"

namespace mytensor {
    template<typename T>
    class Tensor;

    class MatMulNode final : public AutogradNode<float> {
    public:
        MatMulNode(Tensor<float> *intput_a,
                Tensor<float> *input_b);

        static const char *name() { return "AddNode"; }

        void backward(const Tensor<float> &grad_output) override;

    private:
        Tensor<float> *a_;
        Tensor<float> *b_;
    };
}


#endif //MYTENSOR_MATMULNODE_H
