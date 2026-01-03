//
// Created by Mingjie.Zhu22 on 2026/1/3.
//

#ifndef MYTENSOR_KERNALS_H
#define MYTENSOR_KERNALS_H
#include "mytensor/tensor.h"


namespace mytensor::ops {
    template<typename T>
    void add_kernel(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) {


        const T *a_ptr = a.data();
        const T *b_ptr = b.data();
        T *out_ptr = out.data();
        for (std::size_t i = 0; i < a.size(); ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
}


#endif //MYTENSOR_KERNALS_H
