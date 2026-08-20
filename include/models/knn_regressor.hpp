#pragma once

#include "core/tensor.hpp"

class KNNRegressor {
private:
    Tensor X_train_;
    Tensor y_train_;
    int k_; // number of nearest neighbours used to average to produce the output
    bool fitted_;

public:
    KNNRegressor();
    KNNRegressor(int k);

    void fit(
        const Tensor &X_train,
        const Tensor &y_train
    );

    Tensor predict(const Tensor &X) const;

    int k() const;
};