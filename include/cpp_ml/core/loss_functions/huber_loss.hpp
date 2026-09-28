#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

// Huber(d, e) = 0.5 * e^2 if |e| < d and d(|e| - (1/2)d) otherwise
// where d is delta and e = y_true - y_pred
double huber_loss(
    const Tensor &y_true,
    const Tensor &y_pred,
    const double delta
);

cpp_ml::Tensor huber_loss_gradient(
    const Tensor &y_true,
    const Tensor &y_pred,
    const double delta
);

}