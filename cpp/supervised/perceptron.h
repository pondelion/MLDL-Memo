#ifndef PERCEPTRON_
#define PERCEPTRON_
#include <string>
#include "../base_model.h"


namespace ML {

class Perceptron : public BaseModel {
 public:
  Perceptron(uint64_t n_epochs=100,
             float eps=0.01) :
      BaseModel::BaseModel(n_epochs),
      eps_(eps),
      is_fitted_(false) {}
  void Fit(const Eigen::MatrixXf& X,
           const Eigen::MatrixXf& y) override;
  Eigen::MatrixXf Predict(const Eigen::MatrixXf& X) override;
  const Eigen::MatrixXf GetWeights() const;
  const std::string GetSeparationLinearEquation() const;

 private:
  Eigen::MatrixXf w_;
  float eps_;
  bool is_fitted_;
};

}

#endif