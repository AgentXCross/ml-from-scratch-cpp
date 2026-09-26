#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

Tensor threshold(
    const Tensor &x, 
    double cutoff = 0.5,
    double upper = 1.0,
    double lower = 0.0
);

}
