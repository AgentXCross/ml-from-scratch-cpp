#include "core/activations/gelu.hpp"

#include <cmath>
#include <stdexcept>

static const double pi = 3.14159265358979323846;

Tensor gelu(const Tensor &x) {
    // GELU(x) = x * Φ(x)
    // or GELU(x) = 0.5 * (1 + erf(x / sqrt(2)))
    if (x.empty()) {
        throw std::invalid_argument("Cannot apply GELU to an empty tensor");
    }

    Tensor phi(x.shape());

    for (int i = 0; i < x.size(); i++) {
        double value = x.at_flat(i);

        phi.at_flat(i) = 0.5 * (1.0 + std::erf(value / std::sqrt(2.0)));
    }

    return phi.elementwise_multiply(x);
}


Tensor gelu_gradient(const Tensor &x) {
    // GELU(x) = x * Φ(x)
    // GELU'(x) = Φ(x) + xΦ'(x)
    // Derivative of the Gaussian CDF is the Gaussian pdf
    // Φ'(x) = (1 / sqrt(2π)) * e^(-x^2 / 2)
    if (x.empty()) {
        throw std::invalid_argument("Cannot compute GELU gradient of an empty tensor");
    }

    Tensor phi_prime(x.shape());
    Tensor phi(x.shape());

    for (int i = 0; i < x.size(); i++) {
        double value = x.at_flat(i);

        phi_prime.at_flat(i) = (1.0 / std::sqrt(2.0 * pi)) * 
                                std::exp(-0.5 * value * value);
        
        phi.at_flat(i) = 0.5 * (1.0 + std::erf(value / std::sqrt(2.0)));
    }

    return phi + x.elementwise_multiply(phi_prime);
}


GELU::GELU() {
    input_ = Tensor();
    has_input_ = false;
}


Tensor GELU::forward(const Tensor &X) {
    input_ = X;
    has_input_ = true;

    return gelu(X);
}


Tensor GELU::backward(const Tensor &dL_dout) {
    if (!has_input_) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (!dL_dout.has_same_shape(input_)) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Tensor dout_dX = gelu_gradient(input_);

    Tensor dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void GELU::step(double learning_rate) {
    (void) learning_rate;
    // GeLU has no learnable parameters
}
