#pragma once

#include "core/tensor.hpp"
#include "core/layers/layer.hpp"

Tensor gelu(const Tensor &x);

Tensor gelu_gradient(const Tensor &x);

class GELU : public Layer {
private:
    Tensor input_;
    bool has_input_;

public:
    GELU();

    Tensor forward(const Tensor &X) override;

    /*
    out = GELU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Tensor backward(const Tensor &dL_dout) override;

    void step(double learning_rate) override;
};
