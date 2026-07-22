#include "core/loss_functions/mse.hpp"

#include <stdexcept>

double mean_squared_error(
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
        throw std::invalid_argument("mean_squared_error expects rank-2 tensors");
    }

    Tensor error = y_pred - y_true;

    return error.square().sum() / y_true.size();
}

Tensor mean_squared_error_gradient(
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
        throw std::invalid_argument("mean_squared_error_gradient expects rank-2 tensors");
    }

    Tensor dL_dpred = (y_pred - y_true) * (2.0 / y_true.size());

    return dL_dpred;
}
