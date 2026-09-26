#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

double mean_absolute_error(
    const Tensor &y_true,
    const Tensor &y_pred
);

Tensor mean_absolute_error_gradient(
    const Tensor &y_true,
    const Tensor &y_pred
);

}
