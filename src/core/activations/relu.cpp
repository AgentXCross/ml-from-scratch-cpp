#include "core/activations/relu.hpp"

#include <algorithm>
#include <stdexcept>


Matrix relu(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = std::max(0.0, x.at(i, j));
        }
    }

    return result;
}


Matrix relu_gradient(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = (x.at(i, j) > 0.0) ? 1.0 : 0.0;
        }
    }

    return result;
}


ReLU::ReLU() {
    input_ = Matrix();
}


Matrix ReLU::forward(const Matrix &X) {
    input_ = X;

    return relu(X);
}


Matrix ReLU::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }

    if (dL_dout.rows() != input_.rows() || dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout must have the same shape as the input");
    }

    Matrix dout_dX = relu_gradient(input_);

    Matrix dL_dX = dL_dout.elementwise_multiply(dout_dX);

    return dL_dX;
}


void ReLU::step(double learning_rate) {
    (void) learning_rate;
    // ReLU has no trainable parameters
}