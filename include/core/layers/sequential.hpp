#pragma once

#include "core/layers/layer.hpp"
#include "core/matrix.hpp"

#include <memory>
#include <vector>

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    Sequential();

    void add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix &X);

    Matrix backward(const Matrix &dL_dout);

    void step(double learning_rate);

    int size() const;
};