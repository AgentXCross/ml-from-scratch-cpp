#include "models/logistic_regression.hpp"
#include "core/activations/sigmoid.hpp"
#include "core/utils/threshold.hpp"

#include <stdexcept>

LogisticRegression::LogisticRegression() {
    weights_ = Matrix();
    bias_ = Matrix();

    weight_gradients_ = Matrix();
    bias_gradients_ = Matrix();
}

LogisticRegression::LogisticRegression(int num_features) {
    weights_ = Matrix(num_features, 1);
    bias_ = Matrix(1, 1);

    weight_gradients_ = Matrix(num_features, 1);
    bias_gradients_ = Matrix(1, 1);
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

Matrix LogisticRegression::predict_probs(const Matrix &X) const {
    Matrix probabilities = predict_probs(X);

    return threshold(probabilities, 0.5);
}

void LogisticRegression::backward(
    const Matrix &X,
    const Matrix &prediction_gradients
) {
    if (X.rows() != prediction_gradients.rows()) {
        throw std::invalid_argument("X and prediction_gradients must have the same number of rows");
    }

    if (prediction_gradients.cols() != 1) {
        throw std::invalid_argument("prediction_gradients must have exactly one column");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix logits = X.matmul(weights_);

    for (int i = 0; i < logits.rows(); i++) {
        logits.at(i, 0) = logits.at(i, 0) + bias_.at(0, 0);
    }
}

