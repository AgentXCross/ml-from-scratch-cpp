#pragma once

#include "core/tensor.hpp"

class LogisticRegression {
private:
    Tensor weights_;
    Tensor bias_;

    Tensor dL_dw_;
    Tensor dL_db_;

public:
    LogisticRegression();
    LogisticRegression(int num_features);

    Tensor predict_probs(const Tensor &X) const; // returns sigmoid probs
    Tensor predict(const Tensor &X) const; // returns class labels {0, 1}

    void backward(const Tensor &X, const Tensor &dL_dpred);

    void step(double learning_rate);

    Tensor weights() const;
    Tensor bias() const;
};
