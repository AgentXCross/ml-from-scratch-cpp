#include "core/activations/tanh.hpp"

#include <cmath>
#include <stdexcept>


Matrix tanh(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double value = x.at(i, j);

            result.at(i, j) = (std::exp(value) - std::exp(-value)) / 
                                (std::exp(value) + std::exp(-value));
        }
    }

    return result;
}


Matrix tanh_gradient(const Matrix &x) {
    Matrix y = tanh(x);

    Matrix ones = Matrix::ones(x.rows(), x.cols());

    return ones - y.elementwise_multiply(y);
}


Tanh::Tanh() {
    input_ = Matrix();
}


Matrix Tanh::forward(const Matrix &X) {
    input_ = X;

    return tanh(X);
}


Matrix Tanh::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (dL_dout.rows() != input_.rows() || dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout must have the same shape as input");
    }

    Matrix dout_dX = tanh_gradient(input_);

    Matrix dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void Tanh::step(double learning_rate) {
    (void) learning_rate;
    // Tanh has no learnable parameters
}