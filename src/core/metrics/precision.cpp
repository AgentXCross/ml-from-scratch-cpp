#include "core/metrics/precision.hpp"

#include <stdexcept>

double precision_score(
    const Matrix &y_true,
    const Matrix &y_pred,
    double positive_class
) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument("y_true and y_pred must have the same dimensions");
    }

    int tp = 0;
    int fp = 0;

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double truth = y_true.at(i, j);
            double pred = y_pred.at(i, j);

            if (pred == positive_class) {
                if (truth == positive_class) {
                    tp++;
                } else if (truth != positive_class) {
                    fp++;
                }
            }
        }
    }

    if (tp + fp == 0.0) {
        return 0.0;
    }

    return static_cast<double> (tp) / (tp + fp);
}