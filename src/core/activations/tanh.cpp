#include "core/activations/tanh.hpp"

#include <cmath>
#include <stdexcept>


Tensor tanh(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute tanh of an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        double value = x.at_flat(i);

        result.at_flat(i) = (std::exp(value) - std::exp(-value)) / 
                            (std::exp(value) + std::exp(-value));
    }

    return result;
}


Tensor tanh_gradient(const Tensor &x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute tanh gradients of an empty tensor");
    }

    Tensor y = tanh(x);

    Tensor ones = Tensor::ones(x.shape());

    return ones - y.elementwise_multiply(y);
}


Tanh::Tanh() {
    input_ = Tensor();
    has_input_ = false;
}


Tensor Tanh::forward(const Tensor &X) {
    input_ = X;
    has_input_ = true;

    return tanh(X);
}


Tensor Tanh::backward(const Tensor &dL_dout) {
    if (!has_input_) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (!dL_dout.has_same_shape(input_)) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Tensor dout_dX = tanh_gradient(input_);

    Tensor dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void Tanh::step(double learning_rate) {
    (void) learning_rate;
    // Tanh has no learnable parameters
}
