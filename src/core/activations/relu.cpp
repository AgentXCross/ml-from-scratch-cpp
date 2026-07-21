#include "core/activations/relu.hpp"

#include <algorithm>
#include <stdexcept>


Tensor relu(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot apply ReLU to an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        result.at_flat(i) = std::max(0.0, x.at_flat(i));
    }

    return result;
}


Tensor relu_gradient(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute ReLU gradients of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        result.at_flat(i) = (x.at_flat(i) > 0.0) ? 1.0 : 0.0;
    }

    return result;
}


ReLU::ReLU() {
    input_ = Tensor();
    has_input_ = false;
}


Tensor ReLU::forward(const Tensor &X) {
    input_ = X;
    has_input_ = true;

    return relu(X);
}


Tensor ReLU::backward(const Tensor &dL_dout) {
    if (!has_input_) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (!dL_dout.has_same_shape(input_)) {
        throw std::invalid_argument("dL_dout must have the same shape as the input");
    }

    Tensor dout_dX = relu_gradient(input_);

    Tensor dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void ReLU::step(double learning_rate) {
    (void) learning_rate;
    // ReLU has no trainable parameters
}
