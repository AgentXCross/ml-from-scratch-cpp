#pragma once

#include "core/tensor.hpp"

class ADALINE {
private:
    Tensor weights_;
    Tensor bias_;

    Tensor dL_dw_;
    Tensor dL_db_;

public:
    ADALINE();
    ADALINE(int num_features);

    Tensor predict_raw(const Tensor &X) const;
    Tensor predict(const Tensor &X) const;

    void backward(
        const Tensor &X,
        const Tensor &dL_dpred
    );

    void step(double learning_rate);

    Tensor weights() const;
    Tensor bias() const;
};