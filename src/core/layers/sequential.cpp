#include "core/layers/sequential.hpp"

#include <stdexcept>


Sequential::Sequential() {
    layers_ = std::vector<std::unique_ptr<Layer>>();
}


void Sequential::add(std::unique_ptr<Layer> layer) {
    if (layer == nullptr) {
        throw std::invalid_argument("layer cannot be nullptr");
    }

    layers_.push_back(std::move(layer));
}


Matrix Sequential::forward(const Matrix &X) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    Matrix output = X;

    for (int i = 0; i < layers_.size(); i++) {
        output = layers_[i]->forward(output);
    }

    return output;
}


Matrix Sequential::backward(const Matrix &dL_dout) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    Matrix backprop_gradient = dL_dout;

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; i--) {
        backprop_gradient = layers_[i]->backward(backprop_gradient);
    }

    return backprop_gradient;
}


void Sequential::step(double learning_rate) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->step(learning_rate);
    }
}


int Sequential::size() const {
    return static_cast<int>(layers_.size());
}