#pragma once

#include "core/tensor.hpp"
#include "core/layers/layer.hpp"

Tensor relu(const Tensor &x);

Tensor relu_gradient(const Tensor &x);

class ReLU : public Layer {
private:
    Tensor input_;
    bool has_input_;

public:
    ReLU();

    Tensor forward(const Tensor &X) override;

    /*
    out = ReLU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Tensor backward(const Tensor &dL_dout) override;

    void step(double learning_rate) override;
};
