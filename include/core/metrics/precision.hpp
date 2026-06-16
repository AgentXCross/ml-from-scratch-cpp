#pragma once

#include "core/matrix.hpp"

double precision_score(
    const Matrix &y_true,
    const Matrix &y_pred,
    double positive_class = 1.0
);