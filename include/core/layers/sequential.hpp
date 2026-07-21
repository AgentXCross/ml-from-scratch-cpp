#pragma once

#include "core/layers/layer.hpp"
#include "core/tensor.hpp"

#include <memory>
#include <vector>

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    Sequential();

    void add(std::unique_ptr<Layer> layer);

    Tensor forward(const Tensor &X);

    Tensor backward(const Tensor &dL_dout);

    void step(double learning_rate);

    int size() const;
};