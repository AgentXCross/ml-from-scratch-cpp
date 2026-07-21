#include "core/activations/leaky_relu.hpp"

#include <stdexcept>

Tensor leaky_relu(const Tensor &x, double alpha) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute Leaky ReLU of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        if (x.at_flat(i) >= 0.0) {
            result.at_flat(i) = x.at_flat(i);
        } else {
            result.at_flat(i) = x.at_flat(i) * alpha;
        }
    }

    return result;
}


Tensor leaky_relu_gradient(const Tensor &x, double alpha) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute Leaky ReLU gradients of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        if (x.at_flat(i) >= 0.0) {
            result.at_flat(i) = 1;
        } else {
            result.at_flat(i) = alpha;
        }
    }

    return result;
}


LeakyReLU::LeakyReLU() {
    input_ = Tensor();
    alpha_ = 0.01;
    has_input_ = false;
}


LeakyReLU::LeakyReLU(double alpha) {
    if (alpha < 0.0) {
        throw std::invalid_argument("alpha cannot be negative");
    }

    input_ = Tensor();
    alpha_ = alpha;
    has_input_ = false;
}


Tensor LeakyReLU::forward(const Tensor &X) {
    input_ = X;
    has_input_ = true;

    return leaky_relu(X, alpha_);
}


Tensor LeakyReLU::backward(const Tensor &dL_dout) {
    if (!has_input_) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (!dL_dout.has_same_shape(input_)) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Tensor dout_dX = leaky_relu_gradient(input_, alpha_);

    Tensor dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void LeakyReLU::step(double learning_rate) {
    (void) learning_rate;
    // LeakyReLU has no learnable parameters
}
