#include "decision_tree.h"

namespace ML {

DecisionTree::CreateTree(const Eigen::MatrixXf& X,
                         const Eigen::MatrixXf& y) {
  
}

void DecisionTree::Fit(const Eigen::MatrixXf& X,
                       const Eigen::MatrixXf& y) {
  n_features_ = X.cols();
  uint64_t n_samples = X.rows();

  if (y.cols() != 1 || n_samples != y.rows()) {
    throw std::length_error("Invalid y shape");
  }
}

}