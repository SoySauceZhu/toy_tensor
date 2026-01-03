//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_ADDNODE_H
#define MYTENSOR_ADDNODE_H

#include <iostream>
#include <memory>
#include <vector>
#include "autograd.h"


namespace mytensor {
    template<typename T>
    class Tensor;

    class AddNode final : public AutogradNode {
    public:
        AddNode(std::shared_ptr<Tensor<float> > intput_a,
                std::shared_ptr<Tensor<float> > input_b);

        static const char *name() { return "AddNode"; }

        void backward(const Tensor<float> &grad_output) override;

    private:
        std::shared_ptr<Tensor<float> > a_;
        std::shared_ptr<Tensor<float> > b_;
    };
}


#endif //MYTENSOR_ADDNODE_H
