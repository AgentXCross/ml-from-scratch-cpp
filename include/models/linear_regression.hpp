#pragma once

#include "core/matrix.hpp"

class LinearRegression {
private:
    Matrix weights_;
    Matrix bias_;

    Matrix weight_gradients_;
    Matrix bias_gradients_;

public:
    LinearRegression();

    LinearRegression(int num_features);

    Matrix predict(const Matrix& X) const;

    void backward(
        const Matrix &X,
        const Matrix &prediction_gradients
    );

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};