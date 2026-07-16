#pragma once

#include "core/matrix.hpp"
#include "core/layers/layer.hpp"

Matrix relu(const Matrix &x);

Matrix relu_gradient(const Matrix &x);

class ReLU : public Layer {
private:
    Matrix input_;

public:
    ReLU();

    Matrix forward(const Matrix &X) override;

    /*
    out = ReLU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;
};