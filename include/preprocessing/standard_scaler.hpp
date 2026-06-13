#pragma once

#include "core/matrix.hpp"

class StandardScaler {
private:
    Matrix means_;
    Matrix stds_;
    bool fitted_;

public:
    StandardScaler();

    void fit(const Matrix &X); // computes one mean and one std per feature/column
    Matrix transform(const Matrix &X) const; // use stored means and stds to normalize
    Matrix fit_transform(const Matrix &X);

    Matrix means() const;
    Matrix stds() const;
};