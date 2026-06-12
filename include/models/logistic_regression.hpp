#pragma once

#include "core/matrix.hpp"

class LogisticRegression {
private:
    Matrix weights_;
    Matrix bias_;

    Matrix dL_dw_;
    Matrix dL_db_;

public:
    LogisticRegression();
    LogisticRegression(int num_features);

    Matrix predict_probs(const Matrix &X) const; // returns sigmoid probs
    Matrix predict(const Matrix &X) const; // returns class labels {0, 1}

    void backward(const Matrix &X, const Matrix &dL_dpred);

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};
