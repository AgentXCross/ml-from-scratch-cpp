#pragma once

#include "core/matrix.hpp"

double mean_squared_error(
    const Matrix &y_true,
    const Matrix &y_pred
);

Matrix mean_squared_error_gradient(
    const Matrix &y_true,
    const Matrix &y_pred
);