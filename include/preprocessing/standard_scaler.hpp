#pragma once

#include "core/tensor.hpp"

class StandardScaler {
private:
    Tensor means_;
    Tensor stds_;
    bool fitted_;

public:
    StandardScaler();

    void fit(const Tensor &X); // computes one mean and one std per feature/column
    Tensor transform(const Tensor &X) const; // use stored means and stds to normalize
    Tensor fit_transform(const Tensor &X);

    Tensor means() const;
    Tensor stds() const;
};
