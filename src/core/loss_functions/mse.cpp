#include "core/loss_functions/mse.hpp"

#include <stdexcept>

double mean_squared_error(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    if (y_true.rows() != y_pred.rows() ||
        y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }
    
    double sum = 0.0;
    int num_elements = y_true.rows() * y_true.cols();

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double error = y_pred.at(i, j) - y_true.at(i, j);
            sum += error * error;
        }
    }

    return sum / num_elements;
}

Matrix mean_squared_error_gradient(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    if (y_true.rows() != y_pred.rows() ||
        y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    Matrix gradient(y_true.rows(), y_true.cols());
    int num_elements = y_true.rows() * y_true.cols();

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            // Derivative of the MSE equation
            gradient.at(i, j) = (2.0 / num_elements) * (y_pred.at(i, j) - y_true.at(i, j));
        }
    }

    return gradient;
}