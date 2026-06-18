#pragma once

#include "core/matrix.hpp"

class ADALINE {
private:
    Matrix weights_;
    Matrix bias_;

    Matrix dL_dw_;
    Matrix dL_db_;

public:
    ADALINE();
    ADALINE(int num_features);

    Matrix predict_raw(const Matrix &X) const;
    Matrix predict(const Matrix &X) const;

    void backward(
        const Matrix &X,
        const Matrix &dL_dpred
    );

    void step(double learning_rate);

    Matrix weights() const;
    Matrix bias() const;
};