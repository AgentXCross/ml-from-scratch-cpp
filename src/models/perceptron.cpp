#include "models/perceptron.hpp"

#include "core/utils/threshold.hpp"

#include <stdexcept>

Perceptron::Perceptron() {
    weights_ = Matrix();
    bias_ = Matrix();
}

Perceptron::Perceptron(int num_features) {
    weights_ = Matrix::random(num_features, 1);
    bias_ = Matrix::random(1, 1);
}   

Matrix Perceptron::predict(const Matrix &X) const {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix scores = X.matmul(weights_);
    
    for (int i = 0; i < scores.rows(); i++) {
        scores.at(i, 0) = scores.at(i, 0) + bias_.at(0, 0);
    }

    return threshold(scores, 0.0);
}

void Perceptron::train_epoch(
    const Matrix &X,
    const Matrix &y_true,
    double learning_rate
) {
    // Make predictions on X
    // Error for each sample is difference between y_true and the prediction
    // Weights are adjusted every sample if the error is not exactly 0
    // When adjusting, add the lr * error * feature_value
    if (X.rows() != y_true.rows()) {
        throw std::invalid_argument("X and y_true must have the same number of rows");
    }

    if (y_true.cols() != 1) {
        throw std::invalid_argument("y_true must have exactly one column");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix predictions = predict(X);

    for (int i = 0; i < X.rows(); i++) {
        double error = y_true.at(i, 0) - predictions.at(i, 0);

        if (error != 0.0) {
            for (int j = 0; j < X.cols(); j++) {
                weights_.at(j, 0) = weights_.at(j, 0) + learning_rate * error * X.at(i, j);
            }

            bias_.at(0, 0) = bias_.at(0, 0) + learning_rate * error;
        }
    }
}

Matrix Perceptron::weights() const {
    return weights_;
}

Matrix Perceptron::bias() const {
    return bias_;
}