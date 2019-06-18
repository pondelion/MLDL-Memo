#include <iostream>
#include <stdexcept>
#include "perceptron.h"


namespace ML {

void Perceptron::Fit(const Eigen::MatrixXf& X,
                     const Eigen::MatrixXf& y) {
  n_features_ = X.cols();
  uint64_t n_samples = X.rows();

  if (y.cols() != 1 || n_samples != y.rows()) {
    throw std::length_error("Invalid y shape");
  }

  w_ = Eigen::MatrixXf::Random(n_features_+1, 1);
  std::cout << w_ << std::endl;

  Eigen::MatrixXf dw;
  Eigen::MatrixXf bias = Eigen::MatrixXf::Ones(n_samples, 1);
  is_fitted_ = true;
  for (int n=0; n < n_epochs_; n++) {
    Eigen::MatrixXf pred;
    pred = Predict(X);
    for (int i=0; i < n_samples; i++) {
      Eigen::MatrixXf X_(X.rows(), X.cols()+bias.cols());
      X_ << X, bias;
      dw = eps_ * pred(i) * X_.row(i).array();
      w_ += dw.transpose();
    }
  }
}

Eigen::MatrixXf Perceptron::Predict(const Eigen::MatrixXf& X) {
  if (!is_fitted_) {
    throw std::runtime_error("Not yet fitted");
  }

  uint64_t n_samples = X.rows();
  Eigen::MatrixXf bias = Eigen::MatrixXf::Ones(n_samples, 1);
  Eigen::MatrixXf X_(X.rows(), X.cols()+bias.cols());

  X_ << X, bias;

  Eigen::MatrixXf pred{X_ * w_};
  for (int i=0; i < X_.rows(); i++) {
    if (pred(i) >= 0) {
      pred(i) = 1;
    } else {
      pred(i) = -1;
    }
  }
  return pred;
}

const Eigen::MatrixXf Perceptron::GetWeights() const {
  return w_;
}

const std::string Perceptron::GetSeparationLinearEquation() const {
  std::string equation{"y = "};
  equation += std::to_string(w_(0, 0)/w_(1, 0));
  equation += "x ";
  if (w_(2, 0)/w_(1, 0) >= 0.0) {
    equation += "+";
  }
  equation += std::to_string(w_(2, 0)/w_(1, 0));
  return equation;
}

}