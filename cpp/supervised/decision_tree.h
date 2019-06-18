#ifndef DECISION_TREE_
#define DECISION_TREE_
#include <string>
#include "../base_model.h"


namespace ML {

class DecisionTree : public BaseModel {
 public:
  DecisionTree(uint64_t n_epochs) :
      BaseModel::BaseModel(n_epochs) {}
  void Fit(const Eigen::MatrixXf& X,
           const Eigen::MatrixXf& y) override;
  Node* CreateTree(const Eigen::MatrixXf& X,
                   const Eigen::MatrixXf& y);
  Eigen::MatrixXf Predict(const Eigen::MatrixXf& X) override;

 private:
  Node root_;
};

struct Node {
  std::string feature_name;
  float threshold;
  Node* left_node;
  Node* right_node;
};

}

#endif
