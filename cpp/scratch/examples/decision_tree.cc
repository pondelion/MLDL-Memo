#include <vector>
#include <iostream>
#include "../supervised/decision_tree/metrics.h"


int main() {
  std::vector<int> x{{1, 2, 3, 3, 3, 4, 4}};
  std::cout << ML::Metrics::EntropyCategorical(x) << std::endl;
  std::cout << ML::Metrics::EntropyContinuous(x) << std::endl;
  std::cout << ML::Metrics::GiniCategorical(x) << std::endl;
  std::cout << ML::Metrics::GiniContinuous(x) << std::endl;
}
