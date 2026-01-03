#include <iostream>
#include "mytensor/tensor.h"
#include "mytensor/ops.h"

int main() {
    std::vector<std::vector<float> > v = {{1, 2, 3}, {2, 3, 4}};
    std::vector<float> w = {1, 2, 3, 2, 3, 4};

    std::cout << v.size() << std::endl;

    mytensor::Tensor<float> a(std::vector<size_t>{2, 3});
    mytensor::Tensor<float> b(std::vector<size_t>{2, 3}, w);

    auto c = a + b;

    std::cout << c << std::endl;

    if (c.grad_fn()) {
        std::cout << c.grad_fn()->name() << std::endl;
    } else {
        std::cout << "grad_fn: <none>" << std::endl;
    }


    std::cout << "----------" << std::endl;

    return 0;
}
