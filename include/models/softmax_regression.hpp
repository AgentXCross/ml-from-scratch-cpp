#pragma once

#include "core/tensor.hpp"

class SoftmaxRegression {
private:
    Tensor weights_;
    Tensor bias_;

    Tensor dL_dw_;
    Tensor dL_db_;

public:
    SoftmaxRegression();
    SoftmaxRegression(int num_features, int num_classes);

    Tensor predict_probs(const Tensor &X) const;
    Tensor predict(const Tensor &X) const;

    void backward(
        const Tensor &X,
        const Tensor &dL_dlogits
    );

    void step(double learning_rate);

    Tensor weights() const;
    Tensor bias() const;
};