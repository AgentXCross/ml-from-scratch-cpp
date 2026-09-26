#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

/*
recall_score returns recall = TP / (TP + FN)
y_true and y_pred must have the same shape;
*/
double recall_score(
    const Tensor &y_true,
    const Tensor &y_pred,
    double positive_class = 1.0
);

}
