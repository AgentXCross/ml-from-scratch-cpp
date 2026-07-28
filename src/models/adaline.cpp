#include "models/adaline.hpp"

#include "core/utils/threshold.hpp"

#include <cassert>
#include <stdexcept>


ADALINE::ADALINE() 
    : weights_(Tensor()),
      bias_(Tensor()),
      dL_dw_(Tensor()),
      dL_db_(Tensor()) {}


static int validate_num_features(int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be positive");
    }

    return num_features;
}


ADALINE::ADALINE(int num_features) 
    : weights_(Tensor::random({validate_num_features(num_features), 1}, -0.01, 0.01)),
      bias_(Tensor::random({1, 1}, -0.01, 0.01)),
      dL_dw_(Tensor({num_features, 1})),
      dL_db_(Tensor({1, 1})) {}


Tensor ADALINE::predict_raw(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Tensor preds = X.matmul(weights_);
    assert(preds.is_matrix());
    assert(preds.rows() == X.rows() && preds.cols() == 1);

    for (int i = 0; i < X.rows(); i++) {
        preds.at(i, 0) = preds.at(i, 0) + bias_.at(0, 0);
    }

    return preds;
}


Tensor ADALINE::predict(const Tensor &X) const {
    Tensor preds = predict_raw(X);

    preds = threshold(preds, 0.0, 1.0, -1.0);

    return preds;
}


void ADALINE::backward(
    const Tensor &X,
    const Tensor &dL_dpred
) {
    // dL_dpred comes from the loss function
    // pred (raw) = w1x1 + w2x2 + ... + b
    // dpred/dw1 = x1
    // dL/dw1 = (dL/dpred) * (dpred/dw1) = (dL/dpred) * x1
    // dL/dw1 is the first component of dL_dw_
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
    assert(dL_dw_.shape() == weights_.shape());

    double bias_gradient = 0.0;

    for (int i = 0; i < dL_dpred.rows(); i++) {
        bias_gradient = bias_gradient + dL_dpred.at(i, 0);
    }

    dL_db_.at(0, 0) = bias_gradient;
    assert(dL_db_.shape() == bias_.shape());
}

void ADALINE::step(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Tensor ADALINE::weights() const {
    return weights_;
}

Tensor ADALINE::bias() const {
    return bias_;
}