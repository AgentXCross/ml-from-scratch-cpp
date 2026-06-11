#include "core/loss_functions/binary_cross_entropy.hpp"

#include <cmath>
#include <stdexcept>

double binary_cross_entropy(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }
}