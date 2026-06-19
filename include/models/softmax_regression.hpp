#pragma once

#include "core/matrix.hpp"

class SoftmaxRegression {
private:
    Matrix weights_;
    Matrix bias_;

    Matrix dL_dw_;
    Matrix dL_db_;

public:
    SoftmaxRegression();
    SoftmaxRegression(int num_features, int num_classes);

    Matrix predict_probs(const Matrix &X) const;
    Matrix predict(const Matrix &X) const;

    void backward(
        const Matrix &X,
        const Matrix &dL_dlogits
    );

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};