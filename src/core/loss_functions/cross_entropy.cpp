#include "core/loss_functions/cross_entropy.hpp"

#include <cmath>
#include <stdexcept>


double cross_entropy(
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
        throw std::invalid_argument("cross_entropy expects rank-2 tensors");
    }

    double epsilon = 1e-15;
    double loss = 0.0;

    for (int i = 0; i < y_true.rows(); i++) {
        for (int j = 0; j < y_true.cols(); j++) {
            double y = y_true.at(i, j);
            double p = y_pred.at(i, j);

            if (p < epsilon) {
                p = epsilon;
            } else if (p > 1.0 - epsilon) {
                p = 1.0 - epsilon;
            }

            loss -= y * std::log(p);
        }
    }

    return loss / y_true.rows();
}


Tensor cross_entropy_gradient(
    const Tensor &y_true,
    const Tensor &y_probs
) {
    if (!y_true.has_same_shape(y_probs)) {
        throw std::invalid_argument("y_true and y_probs must have the same shape");
    }

    if (y_true.empty() || y_probs.empty()) {
        throw std::invalid_argument("Neither y_true nor y_probs can be empty");
    }

    if (!y_true.is_matrix() || !y_probs.is_matrix()) {
        throw std::invalid_argument("cross_entropy_gradient expects rank-2 tensors");
    }

    Tensor dL_dlogits = y_probs - y_true;

    return dL_dlogits / y_true.rows();
}
