#pragma once

#include "core/matrix.hpp"
#include "core/layers/layer.hpp"

Matrix leaky_relu(const Matrix &x, double alpha);

Matrix leaky_relu_gradient(const Matrix &x, double alpha);

class LeakyReLU : public Layer {
private:
    Matrix input_;
    double alpha_;

public:
    LeakyReLU();
    LeakyReLU(double alpha);

    Matrix forward(const Matrix &X) override;

    /*
    out = LeakyReLU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;
};