//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_MULNODE_H
#define MYTENSOR_MULNODE_H
#include "autograd.h"


namespace mytensor {
    template<typename T>
    class Tensor;

    class MulNode final : public AutogradNode {
    public:
        MulNode(std::shared_ptr<Tensor<float> > intput_a,
                std::shared_ptr<Tensor<float> > input_b);

        static const char *name() { return "AddNode"; }

        void backward(const Tensor<float> &grad_output) override;

    private:
        std::shared_ptr<Tensor<float> > a_;
        std::shared_ptr<Tensor<float> > b_;
    };
}



#endif //MYTENSOR_MULNODE_H