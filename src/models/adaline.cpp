#include "models/adaline.hpp"

#include "core/utils/threshold.hpp"

#include <stdexcept>

ADALINE::ADALINE() {
    weights_ = Matrix();
    bias_ = Matrix();

    dL_dw_ = Matrix();
    dL_db_ = Matrix();
}

ADALINE::ADALINE(int num_features) {
    weights_ = Matrix::random(num_features, 1, -0.01, 0.01);
    bias_ = Matrix::random(1, 1, -0.01, 0.01);

    dL_dw_ = Matrix(num_features, 1);
    dL_db_ = Matrix(1, 1);
}

Matrix ADALINE::predict_raw(const Matrix &X) const {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix preds = X.matmul(weights_);

    for (int i = 0; i < X.rows(); i++) {
        preds.at(i, 0) = preds.at(i, 0) + bias_.at(0, 0);
    }

    return preds;
}

Matrix ADALINE::predict(const Matrix &X) const {
    Matrix preds = predict_raw(X);

    preds = threshold(preds, 0.0, 1.0, -1.0);

    return preds;
}

void ADALINE::backward(
    const Matrix &X,
    const Matrix &dL_dpred
) {
    // dL/dpred comes from the loss function
    // pred (raw) = w1x1 + w2x2 + ... + b
    // dpred/dw1 = x1
    // dL/dw1 = (dL/dpred) * (dpred/dw1) = (dL/dpred) * x1
    // dL/dw1 is the first component of dL_dw_
    if (X.rows() != dL_dpred.rows()) {
        throw std::invalid_argument("X and dL_dpred must have the same number of rows");
    }

    if (dL_dpred.cols() != 1) {
        throw std::invalid_argument("dL_dpred must have exactly one column");
    }

    dL_dw_ = X.transpose().matmul(dL_dpred);

    double bias_gradient = 0.0;

    for (int i = 0; i < dL_dpred.rows(); i++) {
        bias_gradient = bias_gradient + dL_dpred.at(i, 0);
    }

    dL_db_.at(0, 0) = bias_gradient;
}

void ADALINE::step(double learning_rate) {
    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Matrix ADALINE::weights() const {
    return weights_;
}

Matrix ADALINE::bias() const {
    return bias_;
}