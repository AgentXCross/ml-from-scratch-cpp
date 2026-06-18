#include "models/logistic_regression.hpp"
#include "core/activations/sigmoid.hpp"
#include "core/utils/threshold.hpp"

#include <stdexcept>

LogisticRegression::LogisticRegression() {
    weights_ = Matrix();
    bias_ = Matrix();

    dL_dw_ = Matrix();
    dL_db_ = Matrix();
}

LogisticRegression::LogisticRegression(int num_features) {
    weights_ = Matrix::random(num_features, 1, -0.01, 0.01);
    bias_ = Matrix::random(1, 1, -0.01, 0.01);

    dL_dw_ = Matrix(num_features, 1);
    dL_db_ = Matrix(1, 1);
}

Matrix LogisticRegression::predict_probs(const Matrix &X) const {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix logits = X.matmul(weights_);
    for (int i = 0; i < logits.rows(); i++) {
        logits.at(i, 0) = logits.at(i, 0) + bias_.at(0, 0);
    }

    return sigmoid(logits);
}

Matrix LogisticRegression::predict(const Matrix &X) const {
    Matrix probabilities = predict_probs(X);

    return threshold(probabilities, 0.5);
}

void LogisticRegression::backward(
    const Matrix &X,
    const Matrix &dL_dpred
) {
    // dL_dpred[i] is the loss gradient with respect to a prediction or ∂L/∂pred.
    // Let logits = w1x1 + .... wnxn
    // pred = sigmoid(logits)
    // ∂pred/∂logits = sigmoid(logits) * (1 - sigmoid(logits)), done using sigmoid_gradient()
    // ∂L/∂logits = (∂L/∂pred) * (∂pred/∂logits)
    
    // ∂logits/∂w1 = x1, ∂logits/∂w2 = x2, ..., ∂logits/∂b = 1
    // ∂L/∂w1 = (∂L/∂logits) * (∂logits/∂w1) = (∂L/∂logits) * x1

    // For a batch, ∂L/∂w = X.T * (∂L/∂logits)
    if (X.rows() != dL_dpred.rows()) {
        throw std::invalid_argument("X and dL_dpred must hathve the same number of rows");
    }

    if (dL_dpred.cols() != 1) {
        throw std::invalid_argument("dL_dpred must have exactly one column");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix logits = X.matmul(weights_);

    for (int i = 0; i < logits.rows(); i++) {
        logits.at(i, 0) = logits.at(i, 0) + bias_.at(0, 0);
    }

    Matrix dpred_dlogits = sigmoid_gradient(logits);
    Matrix dL_dlogits = dL_dpred.elementwise_multiply(dpred_dlogits);

    dL_dw_ = X.transpose().matmul(dL_dlogits);

    double bias_gradient = 0.0;

    for (int i = 0; i < dL_dlogits.rows(); i++) {
        bias_gradient = bias_gradient + dL_dlogits.at(i, 0);
    }

    dL_db_.at(0, 0) = bias_gradient;
}

void LogisticRegression::step(double learning_rate) {
    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Matrix LogisticRegression::weights() const {
    return weights_;
}

Matrix LogisticRegression::bias() const {
    return bias_;
}
