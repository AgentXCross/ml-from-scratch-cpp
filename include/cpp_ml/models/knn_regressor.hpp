#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <cstddef>

namespace cpp_ml {

class KNNRegressor {
private:
    Tensor X_train_;
    Tensor y_train_;
    std::size_t k_; // number of nearest neighbours used to average to produce the output
    bool fitted_;

public:
    KNNRegressor();
    KNNRegressor(std::size_t k);

    void fit(
        const Tensor &X_train,
        const Tensor &y_train
    );

    Tensor predict(const Tensor &X) const;

    std::size_t k() const;
};

}
