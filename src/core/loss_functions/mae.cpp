#include "core/loss_functions/mae.hpp"

#include <stdexcept>


double mean_absolute_error(
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
        throw std::invalid_argument("mean_absolute_error expects rank-2 tensors");
    }

    Tensor error = y_pred - y_true;

    return error.abs().sum() / y_true.size();
}

Tensor mean_absolute_error_gradient(
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
        throw std::invalid_argument("mean_absolute_error_gradient expects rank-2 tensors");
    }

    Tensor dL_dpred(y_true.shape());

    for (int i = 0; i < y_true.size(); i++) {
        double error = y_pred.at_flat(i) - y_true.at_flat(i);

        if (error > 0.0) {
            dL_dpred.at_flat(i) = 1.0 / y_true.size();
        } else if (error < 0.0) {
            dL_dpred.at_flat(i) = -1.0 / y_true.size();
        } else {
            dL_dpred.at_flat(i) = 0.0; // MAE not differentiable at 0
        }
    }

    return dL_dpred;
}
