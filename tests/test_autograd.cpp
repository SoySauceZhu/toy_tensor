#include <memory>
#include <vector>

#include "mytensor/autograd.h"
#include "mytensor/tensor.h"

int main() {
  auto t = std::make_shared<mytensor::Tensor<float>>(std::vector<size_t>{2}, 1.0f);
  auto node = std::make_shared<mytensor::DummyNode>();
  t->set_grad_fn(node);
  return t->is_leaf() ? 1 : 0;
}
