#ifndef METRICS
#define METRICS_
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>


namespace ML {

class Metrics {
 public:
  template <typename T>
  static float EntropyCategorical(const std::vector<T>& x) {
    std::set<T> uniq_vals(x.begin(), x.end());
    auto len = x.size();
    float p;
    float entropy = 0.0;

    for (auto itr = uniq_vals.begin(); itr != uniq_vals.end(); itr++) {
      auto cnt = std::count(x.begin(), x.end(), *itr);
      p = static_cast<float>(cnt) / static_cast<float>(len);
      entropy -= p * log10(p);
    }

    return entropy;
  }

  template <typename T>
  static float EntropyContinuous(const std::vector<T>& x,
                                 uint32_t n_div=40) {
    auto len = x.size();
    float x_max = *std::max_element(x.begin(), x.end());
    float x_min = *std::min_element(x.begin(), x.end());
    float dx = (x_max - x_min) / n_div;
    float p;
    float entropy = 0.0;

    for (auto div_x = x_min; div_x < x_max; div_x += dx) {
      auto cnt = std::count_if(x.begin(),
                               x.end(),
                               [div_x, dx](T v){ return (div_x <= v) && (v <= div_x+dx); });
      p = static_cast<float>(cnt) / static_cast<float>(len);
      if (p != 0) {
        entropy -= p * log10(p);
      }
    }

    return entropy;
  }

  template <typename T>
  static float GiniCategorical(const std::vector<T>& x) {
    std::set<T> uniq_vals(x.begin(), x.end());
    auto len = x.size();
    float p;
    float gini = 1.0;

    for (auto itr = uniq_vals.begin(); itr != uniq_vals.end(); itr++) {
      auto cnt = std::count(x.begin(), x.end(), *itr);
      p = static_cast<float>(cnt) / static_cast<float>(len);
      gini -= p*p;
    }

    return gini;
  }

  template <typename T>
  static float GiniContinuous(const std::vector<T>& x,
                              uint32_t n_div=40) {
    auto len = x.size();
    float x_max = *std::max_element(x.begin(), x.end());
    float x_min = *std::min_element(x.begin(), x.end());
    float dx = (x_max - x_min) / n_div;
    float p;
    float gini = 1.0;

    for (auto div_x = x_min; div_x < x_max; div_x += dx) {
      auto cnt = std::count_if(x.begin(),
                               x.end(),
                               [div_x, dx](T v){ return (div_x <= v) && (v <= div_x+dx); });
      p = static_cast<float>(cnt) / static_cast<float>(len);
      gini -= p*p;
    }

    return gini;
  }
};

}

#endif
