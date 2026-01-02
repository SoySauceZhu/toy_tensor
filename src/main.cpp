#include <iostream>
#include "mytensor/tensor.h"

int main() {
  mytensor::Tensor<float> t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  t.set_requires_grad(false);
  std::cout << t << std::endl;
  return 0;
}
