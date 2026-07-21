#pragma once

#include "core/tensor.hpp"
#include "core/layers/layer.hpp"

Tensor sigmoid(const Tensor &x);

Tensor sigmoid_gradient(const Tensor &x);

class Sigmoid : public Layer {
private:
    Tensor input_;
    bool has_input_;

public:
    Sigmoid();

    Tensor forward(const Tensor &X) override;

    /*
    out = Sigmoid(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Tensor backward(const Tensor &dL_dout) override;

    void step(double learning_rate) override;
};
