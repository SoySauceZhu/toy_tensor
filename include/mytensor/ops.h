#ifndef MYTENSOR_OPS_H
#define MYTENSOR_OPS_H

#include <memory>

#include "mytensor/tensor.h"
#include "mytensor/autograd/autograd.h"

namespace mytensor::ops {

    // Core op APIs (shared_ptr-based graph-friendly style)
    std::shared_ptr<Tensor<float>> add(const std::shared_ptr<Tensor<float>>& a,
                                      const std::shared_ptr<Tensor<float>>& b);

    std::shared_ptr<Tensor<float>> mul(const std::shared_ptr<Tensor<float>>& a,
                                      const std::shared_ptr<Tensor<float>>& b);

    // 2D matrix multiplication only (for WP3)
    std::shared_ptr<Tensor<float>> matmul2d(const std::shared_ptr<Tensor<float>>& a,
                                           const std::shared_ptr<Tensor<float>>& b);

} // namespace mytensor::ops

namespace mytensor {

    // Operator overloads for shared_ptr<Tensor<float>> to enable: auto c = a + b;
    std::shared_ptr<Tensor<float>> operator+(const std::shared_ptr<Tensor<float>>& a,
                                            const std::shared_ptr<Tensor<float>>& b);

    std::shared_ptr<Tensor<float>> operator*(const std::shared_ptr<Tensor<float>>& a,
                                            const std::shared_ptr<Tensor<float>>& b);

    std::shared_ptr<Tensor<float>> operator+=(const std::shared_ptr<Tensor<float>>& a,
                                            const std::shared_ptr<Tensor<float>>& b);
} // namespace mytensor

#endif // MYTENSOR_OPS_H
