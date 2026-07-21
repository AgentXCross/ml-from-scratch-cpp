#pragma once

#include "core/tensor.hpp"


/*
f1_score returns the F1 score = 2 * precision * recall / (precision + recall)
y_true and y_pred must have the same shape
*/
double f1_score(
    const Tensor &y_true,
    const Tensor &y_pred,
    double positive_class = 1.0
);