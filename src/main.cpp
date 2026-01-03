#include <iostream>
#include "mytensor/tensor.h"
#include "mytensor/ops.h"

int main() {
    std::vector<std::vector<float> > v = {{1, 2, 3}, {2, 3, 4}};
    std::vector<float> w = {1, 2, 3, 2, 3, 4};

    std::cout << v.size() << std::endl;

    auto a = make_shared<mytensor::Tensor<float> >(std::vector<size_t>{2, 3});
    auto b = std::make_shared<mytensor::Tensor<float> >(std::vector<size_t>{2, 3}, w);

    auto c = a + b;

    std::cout << *c << std::endl;

    return 0;
}
