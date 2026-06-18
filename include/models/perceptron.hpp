#pragma once

#include "core/matrix.hpp"

class Perceptron {
private:
    Matrix weights_;
    Matrix bias_;

public:
    Perceptron();
    Perceptron(int num_features);

    Matrix predict(const Matrix &X) const;

    void train_epoch(
        const Matrix &X,
        const Matrix &y_true,
        double learning_rate
    );

    Matrix weights() const;
    Matrix bias() const;
};