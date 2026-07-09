#include "core/layers/linear.hpp"

#include <stdexcept>

Linear::Linear() {
    weights_ = Matrix();
    bias_ = Matrix();

    input_ = Matrix();

    dL_dw_ = Matrix();
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

    dL_dw_ = Matrix(in_features, out_features);
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
    if (dL_dout.rows() != i)
}