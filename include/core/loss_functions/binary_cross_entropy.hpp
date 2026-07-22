#pragma once

#include "core/tensor.hpp"

// L = - (y * log(p) + (1 - y) * log(1 - p))
double binary_cross_entropy(
    const Tensor &y_true,
    const Tensor &y_pred
);

// dL_dp = -y/p + (1-y) / (1-p)
Tensor binary_cross_entropy_gradient(
    const Tensor &y_true,
    const Tensor &y_pred
);