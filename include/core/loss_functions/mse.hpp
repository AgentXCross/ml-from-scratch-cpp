#pragma once

#include "core/tensor.hpp"

double mean_squared_error(
    const Tensor &y_true,
    const Tensor &y_pred
);

Tensor mean_squared_error_gradient(
    const Tensor &y_true,
    const Tensor &y_pred
);
