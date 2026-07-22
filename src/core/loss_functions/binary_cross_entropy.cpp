#include "core/loss_functions/binary_cross_entropy.hpp"

#include <cmath>
#include <stdexcept>


double binary_cross_entropy(
    const Tensor &y_true,
    const Tensor &y_pred
) {
    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Neither y_true nor y_pred can be empty");
    }

    if (!y_true.is_matrix() || !y_pred.is_matrix()) {
        throw std::invalid_argument("binary_cross_entropy expects rank-2 tensors");
    }

    double epsilon = 1e-15; // clip the probabilities to prevent extremes when taking log
    double sum = 0.0;
    int num_elements = y_true.size();

    for (int i = 0; i < y_true.size(); i++) {
        double y = y_true.at_flat(i);
        double p = y_pred.at_flat(i);

        if (p < epsilon) {
            p = epsilon;
        } else if (p > 1.0 - epsilon) {
            p = 1.0 - epsilon;
        }
        
        sum += y * std::log(p) + (1.0 - y) * std::log(1 - p);
    }

    return -sum / num_elements;
}


Tensor binary_cross_entropy_gradient(
    const Tensor &y_true,
    const Tensor &y_pred
) {
    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Neither y_true nor y_pred can be empty");
    }

    if (!y_true.is_matrix() || !y_pred.is_matrix()) {
        throw std::invalid_argument("binary_cross_entropy_gradient expects rank-2 tensors");
    }

    double epsilon = 1e-15;
    int num_elements = y_true.size();

    Tensor dL_dpred(y_true.shape());

    for (int i = 0; i < y_true.size(); i++) {
        double y = y_true.at_flat(i);
        double p = y_pred.at_flat(i);

        if (p < epsilon) {
            p = epsilon;
        } else if (p > 1.0 - epsilon) {
            p = 1.0 - epsilon;
        }

        dL_dpred.at_flat(i) = (-(y / p) + ((1.0 - y) / (1.0 - p))) / num_elements;
    }

    return dL_dpred;
}
