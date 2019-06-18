#ifndef BASE_MODEL_
#define BASE_MODEL_
#include "Eigen/Core"


namespace ML {

class BaseModel {
 public:
  BaseModel(uint64_t n_epochs) :
      n_epochs_(n_epochs) {}
  virtual ~BaseModel() {}
  virtual void Fit(const Eigen::MatrixXf& X, 
                   const Eigen::MatrixXf& y) = 0;
  virtual Eigen::MatrixXf Predict(const Eigen::MatrixXf& X) = 0;

 protected:
  uint64_t n_epochs_;
  uint64_t n_features_;
};

}

#endif