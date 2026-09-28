#include "cpp_ml/core/loss_functions/huber_loss.hpp"

#include <cmath>
#include <stdexcept>

namespace cpp_ml {

double huber_loss(
    const Tensor &y_true,
    const Tensor &y_pred,
    const double delta
) {
    if (delta <= 0.0) {
        throw std::invalid_argument("delta parameter must be positive");
    }

    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Neither y_true nor y_pred can be empty");
    }

    if (!y_true.is_matrix() || !y_pred.is_matrix()) {
        throw std::invalid_argument("Huber Loss expects rank-2 tensors");
    }

    double loss = 0.0;

    for (int i = 0; i < y_true.size(); i++) {
        double error = std::abs(y_true.at_flat(i) - y_pred.at_flat(i));

        if (error <= delta) {
            loss += 0.5 * error * error;
        } else {
            loss += delta * (error - 0.5 * delta);
        }
    }

    return loss / y_pred.rows();
}

cpp_ml::Tensor huber_loss_gradient(
    const Tensor &y_true,
    const Tensor &y_pred,
    const double delta
) {
    if (delta <= 0.0) {
        throw std::invalid_argument("delta parameter must be positive");
    }

    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Neither y_true nor y_pred can be empty");
    }

    if (!y_true.is_matrix() || !y_pred.is_matrix()) {
        throw std::invalid_argument("Huber Loss expects rank-2 tensors");
    }

    Tensor dL_dpred{y_true.shape()};

    for (int i = 0; i < y_true.size(); i++) {
        double error = y_pred.at_flat(i) - y_true.at_flat(i);

        if (std::abs(error) <= delta) {
            dL_dpred.at_flat(i) = error;
        } else if (error > delta) {
            dL_dpred.at_flat(i) = delta;
        } else {
            dL_dpred.at_flat(i) = -delta;
        }
    }

    return dL_dpred;
}

}