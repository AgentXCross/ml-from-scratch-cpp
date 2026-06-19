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
}