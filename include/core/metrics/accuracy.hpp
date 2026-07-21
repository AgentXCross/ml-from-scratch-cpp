#pragma once

#include "core/tensor.hpp"


/*
accuracy_score returns # correct / # total for an arbitrary # of classes
y_true and y_pred must have the same shape
*/
double accuracy_score(
    const Tensor &y_true,
    const Tensor &y_pred
);
