#pragma once

#include "core/matrix.hpp"

class MinMaxScaler {
private:
    Matrix mins_;
    Matrix maxs_;
    bool fitted_;

public:
    MinMaxScaler();

    void fit(const Matrix &X);
    Matrix transform(const Matrix &X) const;
    Matrix fit_transform(const Matrix &X);

    Matrix mins() const;
    Matrix maxs() const;
};