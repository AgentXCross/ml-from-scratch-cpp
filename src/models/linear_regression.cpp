#include "models/linear_regression.hpp"

#include <stdexcept>


LinearRegression::LinearRegression() {
    weights_ = Tensor();
    bias_ = Tensor();

    dL_dw_ = Tensor();
    dL_db_ = Tensor();
}


LinearRegression::LinearRegression(int num_features) {
    weights_ = Tensor::random({num_features, 1}, -0.01, 0.01);
    bias_ = Tensor::random({1, 1}, -0.01, 0.01);

    dL_dw_ = Tensor({num_features, 1});
    dL_db_ = Tensor({1, 1});
}


Tensor LinearRegression::predict(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Tensor predictions = X.matmul(weights_);

    for (int i = 0; i < predictions.rows(); i++) {
        predictions.at(i, 0) += bias_.at(0, 0);
    }

    return predictions;
}


void LinearRegression::backward(
    const Tensor &X,
    const Tensor &dL_dpred
) {
    // dL/dpred[i] is the loss gradient with respect to a prediction.
    // Ex: Let pred = w1x1 + w2x2 + b
    // Then dpred/dw1 = x1.
    // By chain rule, dL/dw1 = dL/dpred * dpred/dw1 = (dL/dpred) * x1.
    // dL_dw1 is the first element in dL_dw_.
    if (!X.is_matrix() || !dL_dpred.is_matrix()) {
        throw std::invalid_argument("X and dL_dpred must be rank-2 tensors");
    }

    if (X.rows() != dL_dpred.rows()) {
        throw std::invalid_argument("X and dL_dpred must have the same number of rows");
    }

    if (dL_dpred.cols() != 1) {
        throw std::invalid_argument("dL_dpred must have exactly one column");
    }

    dL_dw_ = X.transpose().matmul(dL_dpred);

    // For the bias, dpred/db = 1, so dL/db is the sum of dL/dpred.
    double bias_gradient = 0.0;

    for (int i = 0; i < dL_dpred.rows(); i++) {
        bias_gradient = bias_gradient + dL_dpred.at(i, 0);
    }

    dL_db_.at(0, 0) = bias_gradient;
}


void LinearRegression::step(double learning_rate) {
    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}


Tensor LinearRegression::weights() const {
    return weights_;
}


Tensor LinearRegression::bias() const {
    return bias_;
}
