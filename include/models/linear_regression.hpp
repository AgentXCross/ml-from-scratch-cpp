#pragma once

#include "core/matrix.hpp"

class LinearRegression {
private:
    Matrix weights_;
    Matrix bias_;

    Matrix dL_dw_;
    Matrix dL_db_;

public:
    LinearRegression();

    LinearRegression(int num_features);

    Matrix predict(const Matrix& X) const;

    void backward(
        const Matrix &X,
        const Matrix &dL_dpred
    );

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};
