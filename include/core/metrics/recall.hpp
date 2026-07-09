#pragma once

#include "core/matrix.hpp"

/*
recall_score returns recall = TP / (TP + FN)
y_true and y_pred must have the same number of rows and columns.
*/
double recall_score(
    const Matrix &y_true,
    const Matrix &y_pred,
    double positive_class = 1.0
);