#pragma once

#include "core/matrix.hpp"
#include "core/layers/layer.hpp"

Matrix tanh(const Matrix &x);

Matrix tanh_gradient(const Matrix &x);

class Tanh : public Layer {
private:
    Matrix input_;

public:
    Tanh();

    Matrix forward(const Matrix &X) override;

    /*
    out = Tanh(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;
};