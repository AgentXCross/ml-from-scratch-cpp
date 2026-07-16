#include "core/activations/leaky_relu.hpp"

#include <stdexcept>

Matrix leaky_relu(const Matrix &x, double alpha) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) > 0.0) {
                result.at(i, j) = x.at(i, j);
            } else {
                result.at(i, j) = x.at(i, j) * alpha;
            }
        }
    }

    return result;
}


Matrix leaky_relu_gradient(const Matrix &x, double alpha) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) > 0.0) {
                result.at(i, j) = 1.0;
            } else {
                result.at(i, j) = alpha;
            }
        }
    }

    return result;
}


LeakyReLU::LeakyReLU() {
    input_ = Matrix();
    alpha_ = 0.01;
}


LeakyReLU::LeakyReLU(double alpha) {
    if (alpha < 0.0) {
        throw std::invalid_argument("alpha cannot be negative");
    }

    input_ = Matrix();
    alpha_ = alpha;
}


Matrix LeakyReLU::forward(const Matrix &X) {
    input_ = X;

    return leaky_relu(X, alpha_);
}


Matrix LeakyReLU::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (dL_dout.rows() != input_.rows() || dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Matrix dout_dX = leaky_relu_gradient(input_, alpha_);

    Matrix dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void LeakyReLU::step(double learning_rate) {
    (void) learning_rate;
    // LeakyReLU has no learnable parameters
}