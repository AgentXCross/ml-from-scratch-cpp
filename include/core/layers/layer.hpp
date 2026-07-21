#pragma once

#include "core/tensor.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    // Methods that must be implemented

    virtual Tensor forward(const Tensor &X) = 0;

    virtual Tensor backward(const Tensor &dL_dout) = 0;

    virtual void step(double learning_rate) = 0;
};