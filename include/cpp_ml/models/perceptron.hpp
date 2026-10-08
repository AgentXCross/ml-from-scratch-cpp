#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <cstddef>

namespace cpp_ml {

class Perceptron {
private:
    Tensor weights_;
    Tensor bias_;

public:
    Perceptron();
    Perceptron(std::size_t num_features);

    Tensor predict(const Tensor &X) const;

    void train_epoch(
        const Tensor &X,
        const Tensor &y_true,
        double learning_rate
    );

    Tensor weights() const;
    Tensor bias() const;
};

}
