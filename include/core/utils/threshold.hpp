#pragma once

#include "core/matrix.hpp"

Matrix threshold(
    const Matrix &x, 
    double cutoff = 0.5,
    double upper = 1.0,
    double lower = 0.0
);
