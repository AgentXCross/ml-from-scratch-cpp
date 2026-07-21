#pragma once

#include "core/tensor.hpp"


/*
precision_score returns the precision = TP / (TP + FP)
y_true and y_pred must have the same shape.
*/
double precision_score(
    const Tensor &y_true,
    const Tensor &y_pred,
    double positive_class = 1.0
);