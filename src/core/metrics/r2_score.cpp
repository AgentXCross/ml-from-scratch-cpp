#include "core/metrics/r2_score.hpp"

#include <stdexcept>

double r2_score(
    const Matrix &y_true,
    const Matrix &y_pred
) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    int num_elements = y_true.rows() * y_true.cols();

    double y_mean = 0.0;

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            y_mean += y_true.at(i, j);
        }
    }

    y_mean = y_mean / num_elements;

    double ss_res = 0.0; // squared residuals
    double ss_tot = 0.0; // square residuals from the mean

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double residual = y_true.at(i, j) - y_pred.at(i, j);
            double total = y_true.at(i, j) - y_mean;

            ss_res += residual * residual;
            ss_tot += total * total;
        }
    }

    if (ss_tot == 0.0) {
        throw std::invalid_argument("R2 score is undefined when y_true has zero variance");
    }

    return 1 - (ss_res / ss_tot);
}
