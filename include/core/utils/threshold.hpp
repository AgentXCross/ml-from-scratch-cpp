#pragma once

#include "core/tensor.hpp"

Tensor threshold(
    const Tensor &x, 
    double cutoff = 0.5,
    double upper = 1.0,
    double lower = 0.0
);
