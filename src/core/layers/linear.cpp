#include "core/layers/linear.hpp"

#include <cassert>
#include <stdexcept>

Linear::Linear() {
    weights_ = Tensor();
    bias_ = Tensor();

    input_ = Tensor();

    dL_dW_ = Tensor();
    dL_db_ = Tensor();
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

    weights_ = Tensor::random({in_features, out_features}, -0.01, 0.01);
    bias_ = Tensor::random({1, out_features}, -0.01, 0.01);

    input_ = Tensor();

    dL_dW_ = Tensor({in_features, out_features});
    dL_db_ = Tensor({1, out_features});
}


Tensor Linear::forward(const Tensor &X) {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match weight rows");
    }

    input_ = X;

    Tensor output = X.matmul(weights_); // output has shape (# of samples, # of features)
    assert(output.rows() == X.rows() && output.cols() == weights_.cols());
    assert(output.ndim() == 2);

    for (int i = 0; i < output.rows(); i++) {
        for (int j = 0; j < output.cols(); j++) {
            output.at(i, j) = output.at(i, j) + bias_.at(0, j);
        }
    }

    return output;
}


Tensor Linear::backward(const Tensor &dL_dout) {
    if (input_.rows() == 0 || input_.cols() == 0) {
        throw std::runtime_error("forward must be called before backward");
    }  

    if (!dL_dout.is_matrix()) {
        throw std::invalid_argument("dL_dout must be a rank-2 tensor");
    }

    // dL_dout has the same shape as the output shape 
    // So same number of rows as the input X and same number of columns as the weights
    if (dL_dout.rows() != input_.rows()) {
        throw std::invalid_argument("dL_dout rows must match input rows");
    }

    if (dL_dout.cols() != weights_.cols()) {
        throw std::invalid_argument("dL_dout cols must match output columns");
    }

    // dL_dW = X.T @ dL_dout
    dL_dW_ = input_.transpose().matmul(dL_dout);

    for (int j = 0; j < dL_dout.cols(); j++) {
        double bias_gradient = 0.0;

        for (int i = 0; i < dL_dout.rows(); i++) {
            bias_gradient += dL_dout.at(i, j);
        }

        dL_db_.at(0, j) = bias_gradient;
    }

    Tensor dL_dX = dL_dout.matmul(weights_.transpose());

    assert(dL_dW_.shape() == weights_.shape());
    assert(dL_db_.shape() == bias_.shape());

    assert(dL_dX.is_matrix());
    assert(dL_dX.rows() == input_.rows());
    assert(dL_dX.cols() == input_.cols());

    return dL_dX;
}


void Linear::step(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must positive.");
    }
    weights_ = weights_ - (dL_dW_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}


Tensor Linear::weights() const {
    return weights_;
}


Tensor Linear::bias() const {
    return bias_;
}