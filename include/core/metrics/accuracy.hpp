#pragma once

#include "core/matrix.hpp"


/*
accuracy_score returns # correct / # total for an arbitrary # of classes
y_true and y_pred must have the same number of rows and columns
*/
double accuracy_score(
    const Matrix &y_true,
    const Matrix &y_pred
);
