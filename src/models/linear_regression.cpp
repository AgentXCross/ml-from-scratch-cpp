#include "models/linear_regression.hpp"

#include <stdexcept>

LinearRegression::LinearRegression() {
    weights_ = Matrix();
    bias_ = Matrix();

    weight_gradients_ = Matrix();
    bias_gradients_ = Matrix();
}

LinearRegression::LinearRegression(int num_features) {
    weights_ = Matrix(num_features, 1);
    bias_ = Matrix(1, 1);

    weight_gradients_ = Matrix(num_features, 1);
    bias_gradients_ = Matrix(1, 1);
}

Matrix LinearRegression::predict(const Matrix &X) const {
    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Matrix predictions = X.matmul(weights_);

    for (int i = 0; i < predictions.rows(); i++) {
        predictions.at(i, 0) = predictions.at(i, 0) + bias_.at(0, 0);
    }

    return predictions;
}

void LinearRegression::backward(
    const Matrix &X,
    const Matrix &prediction_graidents
) {
    if (X.rows() != prediction_graidents.rows()) {
        throw std::invalid_argument("X and prediction_gradients must have the same number of rows");
    }

    if (prediction_gradients.cols() != 1) {
        throw std::invalid_argument("prediction_gradients must have exactly one column");
    }

    weight_gradients_ = X.transpose().matmul(prediction_graidents);

    double bias_gradient = 0.0;

    for (int i = 0; i < prediction_graidents.rows(); i++) {
        bias_gradient = bias_gradient + prediction_graidents.at(i, 0);
    }

    bias_gradients_.at(0, 0) = bias_gradient;
}

void LinearRegression::step(double learning_rate) {
    weights_ = weights_ - (weight_gradients_ * learning_rate);
    bias_ = bias_ - (bias_gradients_ * learning_rate);
}

Matrix LinearRegression::weights() const {
    return weights_;
}

Matrix LinearRegression::bias() const {
    return bias_;
}
