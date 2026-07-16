#include "core/layers/linear.hpp"

#include <stdexcept>

Linear::Linear() {
    weights_ = Matrix();
    bias_ = Matrix();

    input_ = Matrix();

    dL_dW_ = Matrix();
    dL_db_ = Matrix();
}


Linear::Linear(
    int in_features,
    int out_features
) {
    if (in_features <= 0) {
        throw std::invalid_argument("in_features must be positive");
    }

    if (out_features <= 0) {
        throw std::invalid_argument("out_features must be positive");
    }

    weights_ = Matrix::random(in_features, out_features, -0.01, 0.01);
    bias_ = Matrix::random(1, out_features, -0.01, 0.01);

    input_ = Matrix();

    dL_dW_ = Matrix(in_features, out_features);
    dL_db_ = Matrix(1, out_features);
}


Matrix Linear::forward(const Matrix &X) {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match weight rows");
    }

    input_ = X;

    Matrix output = X.matmul(weights_);

    for (int i = 0; i < output.rows(); i++) {
        for (int j = 0; j < output.cols(); j++) {
            output.at(i, j) = output.at(i, j) + bias_.at(0, j);
        }
    }

    return output;
}


Matrix Linear::backward(const Matrix &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }  

    // dL_dout has the same shape as the output shape 
    // So same number of rows as the input X and same number of columns as the weights
    if (dL_dout.rows() != input_.rows()) {
        throw std::invalid_argument("dL_dout rows must match input rows");
    }

    if (dL_dout.cols() != input_.cols()) {
        throw std::invalid_argument("dL_dout cols must match input cols");
    }

    // dL_dW = X.T @ dL_dout
    dL_dW_ = input_.transpose().matmul(dL_dout);

    for (int j = 0; j < dL_dout.cols(); j++) {
        double bias_gradient = 0.0;

        for (int i = 0; i < dL_dout.rows(); i++) {
            bias_gradient = bias_gradient + dL_dout.at(i, j);
        }

        dL_db_.at(0, j) = bias_gradient;
    }

    Matrix dL_dX = dL_dout.matmul(weights_.transpose());

    return dL_dX;
}


void Linear::step(double learning_rate) {
    weights_ = weights_ - (dL_dW_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}


Matrix Linear::weights() const {
    return weights_;
}


Matrix Linear::bias() const {
    return bias_;
}