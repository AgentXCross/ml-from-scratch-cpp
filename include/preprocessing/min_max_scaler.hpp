#pragma once

#include "core/tensor.hpp"

class MinMaxScaler {
private:
    Tensor mins_;
    Tensor maxs_;
    bool fitted_;

public:
    MinMaxScaler();

    void fit(const Tensor &X);
    Tensor transform(const Tensor &X) const;
    Tensor fit_transform(const Tensor &X);

    Tensor mins() const;
    Tensor maxs() const;
};