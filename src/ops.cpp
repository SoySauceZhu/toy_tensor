// src/ops.cpp
#include "mytensor/ops.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mytensor/autograd/addnode.h"
#include "mytensor/autograd/matmulnode.h"
#include "mytensor/autograd/mulnode.h"

namespace mytensor::ops {

static void require_same_shape(const Tensor<float>& a, const Tensor<float>& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
    }
}

static void require_2d(const Tensor<float>& t, const char* name) {
    if (t.shape().size() != 2) {
        throw std::invalid_argument(std::string(name) + ": expected a 2D tensor");
    }
}

std::shared_ptr<Tensor<float>> add(const std::shared_ptr<Tensor<float>>& a,
                                  const std::shared_ptr<Tensor<float>>& b) {
    if (!a || !b) throw std::invalid_argument("add: null tensor pointer");
    require_same_shape(*a, *b, "add");

    auto out = std::make_shared<Tensor<float>>(a->shape(), 0.0f);

    const size_t n = a->numel();
    for (size_t i = 0; i < n; ++i) {
        (*out)[i] = (*a)[i] + (*b)[i];
    }

    const bool req = a->requires_grad() || b->requires_grad();
    out->set_requires_grad(req);

    if (req) {
        // Assumes WP3 provides AddNode(std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>)
        auto node = std::make_shared<AddNode>(a, b);
        out->set_grad_fn(node);
    }
    return out;
}

std::shared_ptr<Tensor<float>> mul(const std::shared_ptr<Tensor<float>>& a,
                                  const std::shared_ptr<Tensor<float>>& b) {
    if (!a || !b) throw std::invalid_argument("mul: null tensor pointer");
    require_same_shape(*a, *b, "mul");

    auto out = std::make_shared<Tensor<float>>(a->shape(), 0.0f);

    const size_t n = a->numel();
    for (size_t i = 0; i < n; ++i) {
        (*out)[i] = (*a)[i] * (*b)[i];
    }

    const bool req = a->requires_grad() || b->requires_grad();
    out->set_requires_grad(req);

    if (req) {
        // Assumes WP3 provides MulNode(std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>)
        auto node = std::make_shared<MulNode>(a, b);
        out->set_grad_fn(node);
    }
    return out;
}

std::shared_ptr<Tensor<float>> matmul2d(const std::shared_ptr<Tensor<float>>& a,
                                       const std::shared_ptr<Tensor<float>>& b) {
    if (!a || !b) throw std::invalid_argument("matmul2d: null tensor pointer");
    require_2d(*a, "matmul2d(a)");
    require_2d(*b, "matmul2d(b)");

    const auto& as = a->shape(); // [m, k]
    const auto& bs = b->shape(); // [k, n]
    const size_t m = as[0];
    const size_t k = as[1];
    const size_t k2 = bs[0];
    const size_t n = bs[1];

    if (k != k2) {
        throw std::invalid_argument("matmul2d: inner dimensions mismatch");
    }

    auto out = std::make_shared<Tensor<float>>(std::vector<size_t>{m, n}, 0.0f);

    // Row-major indexing assumption (consistent with WP1 strides)
    // a(i, p) at flat = i*k + p
    // b(p, j) at flat = p*n + j
    // out(i, j) at flat = i*n + j
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += (*a)[i * k + p] * (*b)[p * n + j];
            }
            (*out)[i * n + j] = sum;
        }
    }

    const bool req = a->requires_grad() || b->requires_grad();
    out->set_requires_grad(req);

    if (req) {
        // Assumes WP3 provides MatMulNode(std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>)
        auto node = std::make_shared<MatMulNode>(a, b);
        out->set_grad_fn(node);
    }
    return out;
}

} // namespace mytensor::ops

namespace mytensor {

std::shared_ptr<Tensor<float>> operator+(const std::shared_ptr<Tensor<float>>& a,
                                        const std::shared_ptr<Tensor<float>>& b) {
    return ops::add(a, b);
}

std::shared_ptr<Tensor<float>> operator*(const std::shared_ptr<Tensor<float>>& a,
                                        const std::shared_ptr<Tensor<float>>& b) {
    return ops::mul(a, b);
}

} // namespace mytensor
