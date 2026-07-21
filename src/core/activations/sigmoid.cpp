#include "core/activations/sigmoid.hpp"

#include <cmath>
#include <stdexcept>


Tensor sigmoid(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute sigmoid of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        result.at_flat(i) = 1.0 / (1.0 + std::exp(-x.at_flat(i)));
    }

    return result;
}


Tensor sigmoid_gradient(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute sigmoid gradients of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        double sig = 1.0 / (1.0 + std::exp(-x.at_flat(i)));

        result.at_flat(i) = sig * (1 - sig);
    }

    return result;
}


Sigmoid::Sigmoid() {
    input_ = Tensor();
    has_input_ = false;
}


Tensor Sigmoid::forward(const Tensor &X) {
    input_ = X;
    has_input_ = true;

    return sigmoid(X);
}


Tensor Sigmoid::backward(const Tensor &dL_dout) {
    if (!has_input_) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (!dL_dout.has_same_shape(input_)) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Tensor dout_dX = sigmoid_gradient(input_);

    Tensor dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void Sigmoid::step(double learning_rate) {
    (void) learning_rate;
    // Sigmoid has no learnable parameters
}
