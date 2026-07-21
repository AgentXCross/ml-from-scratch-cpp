#include "core/metrics/r2_score.hpp"

#include <stdexcept>

double r2_score(
    const Tensor &y_true,
    const Tensor &y_pred
) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("y_true and y_pred cannot be empty");
    }

    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    int num_elements = y_true.size();

    double y_mean = 0.0;

    for (int i = 0; i < y_true.size(); i++) {
        y_mean += y_true.at_flat(i);
    }

    y_mean = y_mean / num_elements;

    double ss_res = 0.0; // squared residuals
    double ss_tot = 0.0; // square residuals from the mean

    for (int i = 0; i < y_true.size(); i++) {
        double residual = y_true.at_flat(i) - y_pred.at_flat(i);
        double total = y_true.at_flat(i) - y_mean;

        ss_res += residual * residual;
        ss_tot += total * total;
    }

    if (ss_tot == 0.0) {
        throw std::invalid_argument("R2 score is undefined when y_true has zero variance");
    }

    return 1 - (ss_res / ss_tot);
}
