#pragma once

#include "core/tensor.hpp"

/*
Coefficient of determination (R^2 score) between the true and
predicted values.

R^2 = 1 - SS_res / SS_tot
    SS_res = Σ(y_true - y_pred)^2
    SS_tot = Σ(y_true - mean(y_true))^2

1.0 -> perfect preds, 0.0 -> preds are no better than just predicting the mean
*/
double r2_score(
    const Tensor &y_true,
    const Tensor &y_pred
);
