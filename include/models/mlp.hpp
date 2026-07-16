#pragma once

#include "core/layers/sequential.hpp"
#include "core/matrix.hpp"

#include <vector>

/*
This version is for binary classification.
*/

class MLP {
private:
    Sequential network_;

public:
    MLP();

    /*
    layer_sizes defines the width of each layer.

    Example:
        {30, 64, 32, 1}

    creates:
        Linear(30, 64) -> ReLU
        Linear(64, 32) -> ReLU
        Linear(32, 1) -> Sigmoid
    */
    MLP(const std::vector<int> &layer_sizes);

    Matrix forward(const Matrix &X);

    Matrix backward(const Matrix &dL_dout);

    void step(double learning_rate);

    Matrix predict(const Matrix &X);
};