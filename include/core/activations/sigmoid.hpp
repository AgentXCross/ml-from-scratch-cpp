#pragma once

#include "core/matrix.hpp"
#include "core/layers/layer.hpp"

Matrix sigmoid(const Matrix &x);

Matrix sigmoid_gradient(const Matrix &x);

class Sigmoid : public Layer {
private:
    Matrix input_;

public:
    Sigmoid();

    Matrix forward(const Matrix &X) override;

    /*
    out = Sigmoid(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;
};