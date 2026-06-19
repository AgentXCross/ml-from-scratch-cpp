#include "core/loss_functions/cross_entropy.hpp"

#include <cmath>
#include <stdexcept>

double cross_entropy(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    // CE = -Σ(y * log(p)) where y is the truth for the class and p is the probability of the class
    if (y_true.rows() != y_pred.rows() ||
        y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    double epsilon = 1e-15;
    double sum = 0.0;

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double y = y_true.at(i, j);
            double p = y_pred.at(i, j);

            if (p < epsilon) {
                p = epsilon;
            } else if (p > 1.0 - epsilon) {
                p = 1.0 - epsilon;
            }

            sum = sum + y * std::log(p);
        }
    }

    return -sum / y_true.rows();
}

Matrix cross_entropy_gradient(
    const Matrix &y_true,
    const Matrix &y_probs
) {
    // This returns the combined softmax + cross entropy gradient
    // CE = -Σ(y * log(p))
    // p = softmax(logits)
    // Derivative of softmax and cross entropy simplify to
    // dL/dlogits = p - y
    if (y_true.rows() != y_probs.rows() ||
        y_true.cols() != y_probs.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    Matrix dL_dlogits(y_true.rows(), y_true.cols());

    int num_samples = y_true.rows();

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            dL_dlogits.at(i, j) = (y_probs.at(i, j) - y_true.at(i, j)) / num_samples;
        }
    }

    return dL_dlogits;
}