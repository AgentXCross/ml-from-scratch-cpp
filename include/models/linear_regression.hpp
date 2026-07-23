#pragma once

#include "core/tensor.hpp"

class LinearRegression {
private:
    Tensor weights_;
    Tensor bias_;

    Tensor dL_dw_;
    Tensor dL_db_;

public:
    LinearRegression();

    LinearRegression(int num_features);

    Tensor predict(const Tensor &X) const;

    void backward(
        const Tensor &X,
        const Tensor &dL_dpred
    );

    void step(double learning_rate);

    Tensor weights() const;
    Tensor bias() const;
};
