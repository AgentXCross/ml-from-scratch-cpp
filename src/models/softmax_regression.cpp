#include "models/softmax_regression.hpp"

#include "core/activations/softmax.hpp"
#include "core/utils/argmax.hpp"

#include <stdexcept>

SoftmaxRegression::SoftmaxRegression() {
    weights_ = Matrix();
    bias_ = Matrix();

    dL_dw_ = Matrix();
    dL_db_ = Matrix();
}

SoftmaxRegression::SoftmaxRegression(int num_features, int num_classes) {
    if (num_features <= 0) {
        throw std::invalid_argument("Must have at least one feature");
    }
    
    if (num_classes <= 1) {
        throw std::invalid_argument("Must have more than one class");
    }

    // weights are a 2D matrix
    // Each feature has a row and each column is for the classes
    weights_ = Matrix::random(num_features, num_classes, -0.01, 0.01);
    bias_ = Matrix::random(1, num_classes, -0.01, 0.01);

    dL_dw_ = Matrix(num_features, num_classes);
    dL_db_ = Matrix(1, num_classes);
}

Matrix SoftmaxRegression::predict_probs(const Matrix &X) const {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weight rows");
    }

    Matrix logits = X.matmul(weights_);

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < logits.cols(); j++) {
            logits.at(i, j) = logits.at(i, j) + bias_.at(0, j);
        }
    }

    return softmax(logits);
}

Matrix SoftmaxRegression::predict(const Matrix &X) const {
    Matrix probabilities = predict_probs(X);
    Matrix predictions(probabilities.rows(), 1);

    for (int i = 0; i < predictions.rows(); i++) {
        Matrix row = probabilities.row(i);
        int predicted_class = argmax(row);

        predictions.at(i, 0) = static_cast<double> (predicted_class);
    }

    return predictions;
}

void SoftmaxRegression::backward(
    const Matrix &X,
    const Matrix &dL_dlogits
) {
    // dL_dlogits comes from cross_entropy_gradient
    // logits = XW + b, where X is num_samples x num_features, W is num_features x num_classes
    // dL_dW = X.T @ dL_dlogits
    if (X.rows() != dL_dlogits.rows()) {
        throw std::invalid_argument("X and dL_dlogits must have the same number of rows");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weight rows");
    }

    if (dL_dlogits.cols() != weights_.cols()) {
        throw std::invalid_argument("dL_dlogits columns must match number of classes");
    }

    dL_dw_ = X.transpose().matmul(dL_dlogits);

    for (int j = 0; j < dL_dlogits.cols(); j++) {
        double bias_gradient = 0.0;

        for (int i = 0; i < dL_dlogits.rows(); i++) {
            bias_gradient = bias_gradient + dL_dlogits.at(i, j);
        }

        dL_db_.at(0, j) = bias_gradient;
    }
}

void SoftmaxRegression::step(double learning_rate) {
    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Matrix SoftmaxRegression::weights() const {
    return weights_;
}

Matrix SoftmaxRegression::bias() const {
    return bias_;
}