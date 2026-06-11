#pragma once

#include "core/matrix.hpp"

double binary_cross_entropy(
    const Matrix &y_true,
    const Matrix &y_pred
);

Matrix binary_cross_entropy_gradient(
    const Matrix &y_true,
    const Matrix &y_pred
);