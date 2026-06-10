#include "core/metrics/accuracy.hpp"

#include <stdexcept>

double accuracy_score(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    int num_elements = y_true.rows() * y_true.cols();
    int num_correct = 0;

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            if (y_true.at(i, j) == y_pred.at(i, j)) {
                num_correct++;
            }
        }
    }

    return static_cast<double> (num_correct) / num_elements;
}