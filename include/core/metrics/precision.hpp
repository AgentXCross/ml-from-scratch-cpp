#pragma once

#include "core/matrix.hpp"


/*
precision_score returns the precision = TP / (TP + FP)
y_true and y_pred must have the same number of columns and rows.
*/
double precision_score(
    const Matrix &y_true,
    const Matrix &y_pred,
    double positive_class = 1.0
);