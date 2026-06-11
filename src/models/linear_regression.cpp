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
    const Matrix &prediction_gradients
) {
    // prediction_gradients[i] = ∂L/∂pred
    // Ex: Let pred = w1x1 + w2x2 + b
    // Then ∂pred/∂w1 = x1
    // By the Chain Rule, ∂L/∂w1 = ∂L/∂pred * ∂pred/∂w1 = (∂L/∂pred) * x1
    // ∂L/∂w1 is the first element in weight_gradients_
    if (X.rows() != prediction_gradients.rows()) {
        throw std::invalid_argument("X and prediction_gradients must have the same number of rows");
    }

    if (prediction_gradients.cols() != 1) {
        throw std::invalid_argument("prediction_gradients must have exactly one column");
    }

    weight_gradients_ = X.transpose().matmul(prediction_gradients);

    // From the example above, taking the partial deriative of the prediction with respect to the bias
    // ∂pred/∂b = 1
    // So ∂L/∂b = ∂L/∂pred = prediction_gradients[i]
    double bias_gradient = 0.0;

    for (int i = 0; i < prediction_gradients.rows(); i++) {
        bias_gradient = bias_gradient + prediction_gradients.at(i, 0);
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
