#pragma once

#include "core/tensor.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    // Methods that must be implemented

    /*
    Forward Pass.
    Input to this layer is stored within the class, before computing and passing 
    the output into the subsequent layer.
    */
    virtual Tensor forward(const Tensor &X) = 0;

    /*
    Backpropagation.
    Receives the gradient of the loss with respect to this layers output (dL_dout). 
    Computes gradients with respect to this layers parameters (dL_dw) and input (dL_dx).
    Returns the gradient with respect to the the previous layers output (dL_dx becomes the dL_dout of the previous layer).
    */
    virtual Tensor backward(const Tensor &dL_dout) = 0;

    /*
    Gradient Descent.
    Updates this layers learnable parameters using the stored gradients from backward.
        parameter = parameter - learning_rate * gradient
    The gradient tensor must have the same shape as the learnable parameters.
    */
    virtual void step(double learning_rate) = 0;
};