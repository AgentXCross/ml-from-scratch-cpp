#pragma once

#include "core/matrix.hpp"

class Linear {
private:
    Matrix weights_; // Shape: in_features x out_features
    Matrix bias_; // Shape: 1 x out_features

    Matrix input_;

    Matrix dL_dw_;
    Matrix dL_db_;

public:
    Linear();
    Linear(
        int in_features,
        int out_features
    );

    Matrix forward(const Matrix &X); // out = XW + b

    Matrix backward(const Matrix &dL_dout); // returns dL_dX, dL_dout has the same shape as the output of the layer

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};