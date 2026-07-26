#include "models/logistic_regression.hpp"

#include "core/activations/sigmoid.hpp"
#include "core/utils/threshold.hpp"

#include <cassert>
#include <stdexcept>

LogisticRegression::LogisticRegression()
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


LogisticRegression::LogisticRegression(int num_features) 
    : weights_(Tensor::random({validate_num_features(num_features), 1}, -0.01, 0.01)),
      bias_(Tensor::random({1, 1}, -0.01, 0.01)),
      dL_dw_(Tensor({num_features, 1})),
      dL_db_(Tensor({1, 1})) {}


Tensor LogisticRegression::predict_probs(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Tensor logits = X.matmul(weights_);
    assert(logits.is_matrix());
    assert(logits.cols() == 1);

    for (int i = 0; i < logits.rows(); i++) {
        logits.at(i, 0) = logits.at(i, 0) + bias_.at(0, 0);
    }

    return sigmoid(logits);
}

Tensor LogisticRegression::predict(const Tensor &X) const {
    Tensor probabilities = predict_probs(X);

    Tensor predictions = threshold(probabilities, 0.5);
    assert(probabilities.shape() == predictions.shape());

    return predictions;
}

void LogisticRegression::backward(
    const Tensor &X,
    const Tensor &dL_dpred
) {
    // dL_dpred[i] is the loss gradient with respect to a prediction or dL_dpred.
    // Let logits = w1x1 + .... wnxn
    // pred = sigmoid(logits)
    // dpred_dlogits = sigmoid(logits) * (1 - sigmoid(logits)), done using sigmoid_gradient()
    // dL_dlogits = (dL_dpred) * (dpred_dlogits)
    
    // dlogits_dw1 = x1, dlogits_dw2 = x2, ..., dlogits_db = 1
    // dL_dw1 = (dL_dlogits) * (dlogits_dw1) = (dL_dlogits) * x1

    // For a batch, dL_dw = X.T * (dL_dlogits)
    if (!X.is_matrix() || !dL_dpred.is_matrix()) {
        throw std::invalid_argument("X and dL_dpred must both be rank-2 tensors");
    }

    if (X.rows() != dL_dpred.rows()) {
        throw std::invalid_argument("X and dL_dpred must have the same number of rows");
    }

    if (dL_dpred.cols() != 1) {
        throw std::invalid_argument("dL_dpred must have exactly one column");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    // Slightly wasteful as we need to recalculate logits, consider caching later
    Tensor logits = X.matmul(weights_);
    assert(logits.is_matrix());
    assert(logits.cols() == 1);

    for (int i = 0; i < logits.rows(); i++) {
        logits.at(i, 0) = logits.at(i, 0) + bias_.at(0, 0);
    }

    Tensor dpred_dlogits = sigmoid_gradient(logits);
    Tensor dL_dlogits = dL_dpred.elementwise_multiply(dpred_dlogits);
    assert(logits.shape() == dL_dlogits.shape());

    dL_dw_ = X.transpose().matmul(dL_dlogits);
    assert(dL_dw_.shape() == weights_.shape());

    double bias_gradient = 0.0;

    for (int i = 0; i < dL_dlogits.rows(); i++) {
        bias_gradient = bias_gradient + dL_dlogits.at(i, 0);
    }

    dL_db_.at(0, 0) = bias_gradient;
    assert(dL_db_.shape() == bias_.shape());
}

void LogisticRegression::step(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }
    
    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Tensor LogisticRegression::weights() const {
    return weights_;
}

Tensor LogisticRegression::bias() const {
    return bias_;
}
