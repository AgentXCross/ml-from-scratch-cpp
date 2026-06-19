#pragma once

#include "core/matrix.hpp"

double cross_entropy(
    const Matrix &y_true,
    const Matrix &y_pred
);

Matrix cross_entropy_gradient(
    const Matrix &y_true,
    const Matrix &y_probs
);