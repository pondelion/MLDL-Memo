#include <iostream>
#include "Eigen/Core"
#include "../supervised/perceptron.h"


int main() {
  constexpr uint64_t N_SAMPLES = 100;
  ML::Perceptron perceptron_model;
  Eigen::MatrixXf X1 = Eigen::MatrixXf::Random(N_SAMPLES, 2);
  Eigen::MatrixXf X2 = Eigen::MatrixXf::Random(N_SAMPLES, 2);
  Eigen::MatrixXf y1 = Eigen::MatrixXf::Ones(N_SAMPLES, 1);
  Eigen::MatrixXf y2 = Eigen::MatrixXf::Ones(N_SAMPLES, 1).array()-2;
  X1 = X1.array()+3;
  X2 = X2.array()-2;
  Eigen::MatrixXf X(X1.rows()+X2.rows(), X1.cols());
  Eigen::MatrixXf y(y1.rows()+y2.rows(), y1.cols());
  X << X1, X2;
  y << y1, y2;
  std::cout << X << std::endl;
  //std::cout << perceptron_model.Predict(X) << std::endl;
  perceptron_model.Fit(X, y);
  std::cout << perceptron_model.Predict(X) << std::endl;
  std::cout << perceptron_model.GetWeights() << std::endl;
  std::cout << perceptron_model.GetSeparationLinearEquation() << std::endl;
}