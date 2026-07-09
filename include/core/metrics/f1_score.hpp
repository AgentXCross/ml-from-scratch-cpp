#pragma once

#include "core/matrix.hpp"


/*
f1_score returns the F1 score = 2 * precision * recall / (precision + recall)
y_true and y_pred must have the same number of columns and rows.
*/
double f1_score(
    const Matrix &y_true,
    const Matrix &y_pred,
    double positive_class = 1.0
);