#include "core/layers/sequential.hpp"

#include <cassert>
#include <stdexcept>


Sequential::Sequential()
    : layers_(std::vector<std::unique_ptr<Layer>>()) {}


void Sequential::add(std::unique_ptr<Layer> layer) {
    if (layer == nullptr) {
        throw std::invalid_argument("layer cannot be nullptr");
    }

    layers_.push_back(std::move(layer));
}


Tensor Sequential::forward(const Tensor &X) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    Tensor output = X;

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        output = layers_[i]->forward(output);
        assert(!output.empty());
    }

    return output;
}


Tensor Sequential::backward(const Tensor &dL_dout) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    if (dL_dout.empty()) {
        throw std::invalid_argument("dL_dout cannot be empty");
    }

    Tensor backprop_gradient = dL_dout;

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; i--) {
        backprop_gradient = layers_[i]->backward(backprop_gradient);
        assert(!backprop_gradient.empty());
    }

    return backprop_gradient;
}


void Sequential::step(double learning_rate) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        layers_[i]->step(learning_rate);
    }
}


int Sequential::size() const {
    return static_cast<int>(layers_.size());
}
