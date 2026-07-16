#include "core/activations/gelu.hpp"

#include <cmath>
#include <stdexcept>

static const double pi = 3.14159265358979323846;

Matrix gelu(const Matrix &x) {
    // GELU(x) = x * Φ(x)
    // or GELU(x) = 0.5 * (1 + erf(x / sqrt(2)))
    Matrix phi(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            phi.at(i, j) = 0.5 * (1 + std::erf(x.at(i, j) / std::sqrt(2)));
        }
    }

    return phi.elementwise_multiply(x);
}


Matrix gelu_gradient(const Matrix &x) {
    // GELU(x) = x * Φ(x)
    // GELU'(x) = Φ(x) + xΦ'(x)
    // Derivative of the Gaussian CDF is the Gaussian pdf
    // Φ'(x) = (1 / sqrt(2π)) * e^(-x^2 / 2)
    Matrix phi_prime(x.rows(), x.cols());
    Matrix phi(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            phi_prime.at(i, j) = (1 / std::sqrt(2 * pi)) * 
                                    std::exp(- 0.5 * (x.at(i, j) * x.at(i, j)));
        
            phi.at(i, j) = 0.5 * (1 + std::erf(x.at(i, j) / std::sqrt(2)));
        }
    }

    return phi + x.elementwise_multiply(phi_prime);
}


GELU::GELU() {
    input_ = Matrix();
}


Matrix GELU::forward(const Matrix &X) {
    input_ = X;

    return gelu(X);
}


Matrix GELU::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (dL_dout.rows() != input_.rows() || dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Matrix dout_dX = gelu_gradient(input_);

    Matrix dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void GELU::step(double learning_rate) {
    (void) learning_rate;
    // GeLU has no learnable parameters
}