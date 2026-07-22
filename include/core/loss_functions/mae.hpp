#pragma once

#include "core/tensor.hpp"

double mean_absolute_error(
    const Tensor &y_true,
    const Tensor &y_pred
);

Tensor mean_absolute_error_gradient(
    const Tensor &y_true,
    const Tensor &y_pred
);