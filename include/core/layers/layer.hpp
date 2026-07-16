#pragma once

#include "core/matrix.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    // Methods that must be implemented

    virtual Matrix forward(const Matrix &X) = 0;

    virtual Matrix backward(const Matrix &dL_dout) = 0;

    virtual void step(double learning_rate) = 0;
};