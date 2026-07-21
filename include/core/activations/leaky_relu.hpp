#pragma once

#include "core/tensor.hpp"
#include "core/layers/layer.hpp"

Tensor leaky_relu(const Tensor &x, double alpha);

Tensor leaky_relu_gradient(const Tensor &x, double alpha);

class LeakyReLU : public Layer {
private:
    Tensor input_;
    double alpha_;
    bool has_input_;

public:
    LeakyReLU();
    LeakyReLU(double alpha);

    Tensor forward(const Tensor &X) override;

    /*
    out = LeakyReLU(X)
    dL_dX = dL_dout * dout_dX

    Returns dL_dX
    */
    Tensor backward(const Tensor &dL_dout) override;

    void step(double learning_rate) override;
};
