#include "core/loss_functions/binary_cross_entropy.hpp"

#include <cmath>
#include <stdexcept>

double binary_cross_entropy(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    // L = - (y * log(p) + (1 - y) * log(1 - p))
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    double epsilon = 1e-15; // clip the probabilities to prevent extremes when taking log
    double sum = 0.0;
    int num_elements = y_true.rows() * y_true.cols();

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double y = y_true.at(i, j);
            double p = y_pred.at(i, j);

            if (p < epsilon) {
                p = epsilon;
            } else if (p > 1.0 - epsilon) {
                p = 1.0 - epsilon;
            }
            
            sum += y * std::log(p) + (1.0 - y) * std::log(1 - p);
        }
    }

    return -sum / num_elements;
}

Matrix binary_cross_entropy_gradient(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    // ∂L/∂p = -y/p + (1-y)/(1-p)
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_cols must have the same dimensions");
    }

    double epsilon = 1e-15;
    int num_elements = y_true.rows() * y_true.cols();

    Matrix gradient(y_true.rows(), y_true.cols());

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double y = y_true.at(i, j);
            double p = y_pred.at(i, j);

            if (p < epsilon) {
                p = epsilon;
            } else if (p > 1.0 - epsilon) {
                p = 1.0 - epsilon;
            }

            gradient.at(i, j) = (-(y / p) + ((1.0 - y) / (1.0 - p))) / num_elements;
        }
    }

    return gradient;
}