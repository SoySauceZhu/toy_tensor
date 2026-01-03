//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_ADDNODE_H
#define MYTENSOR_ADDNODE_H

#include <memory>
#include <utility>
#include "autograd.h"

namespace mytensor {
    template<typename T>
    class Tensor;

    template<typename T>
    class AddNode final : public AutogradNode<T> {
    public:
        AddNode(std::shared_ptr<Tensor<T> > intput_a,
                std::shared_ptr<Tensor<T> > input_b)
            : a_(std::move(intput_a)), b_(std::move(input_b)) {
            this->inputs.push_back(a_);
            this->inputs.push_back(b_);
        }

        static const char *name() { return "AddNode"; }

        void backward(const Tensor<T> &grad_output) {
        }

    private:
        std::shared_ptr<Tensor<T> > a_;
        std::shared_ptr<Tensor<T> > b_;
    };
}


#endif //MYTENSOR_ADDNODE_H
