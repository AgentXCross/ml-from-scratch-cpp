#include "core/activations/leaky_relu.hpp"

Matrix leaky_relu(const Matrix &x, double alpha) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) > 0.0) {
                result.at(i, j) = x.at(i, j);
            } else {
                result.at(i, j) = x.at(i, j) * alpha;
            }
        }
    }

    return result;
}

Matrix leaky_relu_gradient(const Matrix &x, double alpha) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) > 0.0) {
                result.at(i, j) = 1.0;
            } else {
                result.at(i, j) = alpha;
            }
        }
    }

    return result;
}