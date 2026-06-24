#pragma once

#include "core/matrix.hpp"

Matrix one_hot_encode(
    const Matrix &y,
    int num_classes
);