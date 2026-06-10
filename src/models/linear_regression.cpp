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

    Matrix predictions = X.matmul(weights);

    for (int i = 0; i < predictions.rows(); i++) {
        predictions.at(i, 0) = predictions.at(i, 0) + bias_.at(0, 0);
    }

    return predictions;
}

