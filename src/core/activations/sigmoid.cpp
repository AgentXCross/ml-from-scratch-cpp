#include "core/activations/sigmoid.hpp"

#include <cmath>
#include <stdexcept>


Matrix sigmoid(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = 1.0 / (1.0 + std::exp(-x.at(i, j)));
        }
    }

    return result;
}


Matrix sigmoid_gradient(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double sig = 1.0 / (1.0 + std::exp(-x.at(i, j)));

            result.at(i, j) = sig * (1 - sig);
        }
    }

    return result;
}


Sigmoid::Sigmoid() {
    input_ = Matrix();
}


Matrix Sigmoid::forward(const Matrix &X) {
    input_ = X;

    return sigmoid(X);
}


Matrix Sigmoid::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (dL_dout.rows() != input_.rows() || dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Matrix dout_dX = sigmoid_gradient(input_);

    Matrix dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void Sigmoid::step(double learning_rate) {
    (void) learning_rate;
    // Sigmoid has no learnable parameters
}