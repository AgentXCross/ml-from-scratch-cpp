#pragma once

#include "core/tensor.hpp"
#include "core/layers/layer.hpp"

Tensor tanh(const Tensor &x);

Tensor tanh_gradient(const Tensor &x);

class Tanh : public Layer {
private:
    Tensor input_;
    bool has_input_;

public:
    Tanh();

    Tensor forward(const Tensor &X) override;

    /*
    out = Tanh(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Tensor backward(const Tensor &dL_dout) override;

    void step(double learning_rate) override;
};
