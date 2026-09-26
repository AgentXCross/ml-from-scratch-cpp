#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

class KNN {
private:
    Tensor X_train_;
    Tensor y_train_;
    int k_; // # of neighbors considered in classifying
    bool fitted_;

public:
    KNN();
    KNN(int k);

    void fit(
        const Tensor &X_train,
        const Tensor &y_train
    );

    Tensor predict(const Tensor &X) const;

    int k() const;
};

}
