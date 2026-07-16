#pragma once

#include "core/matrix.hpp"
#include "core/layers/layer.hpp"

Matrix gelu(const Matrix &x);

Matrix gelu_gradient(const Matrix &x);

class GELU : public Layer {
private:
    Matrix input_;

public:
    GELU();

    Matrix forward(const Matrix &X) override;

    /*
    out = GeLU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;
};