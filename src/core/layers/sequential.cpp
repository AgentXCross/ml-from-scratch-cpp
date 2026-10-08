#include "cpp_ml/core/layers/sequential.hpp"

#include <cassert>
#include <stdexcept>

namespace cpp_ml {

Sequential::Sequential()
    : layers_{} {}


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

    Tensor output{X};

    for (std::size_t i{0}; i < layers_.size(); ++i) {
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

    Tensor backprop_gradient{dL_dout};

    for (std::size_t i{layers_.size()}; i-- > 0;) {
        backprop_gradient = layers_[i]->backward(backprop_gradient);
        assert(!backprop_gradient.empty());
    }

    return backprop_gradient;
}


void Sequential::step(double learning_rate) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential must have at least one layer");
    }

    for (std::size_t i{0}; i < layers_.size(); ++i) {
        layers_[i]->step(learning_rate);
    }
}


std::size_t Sequential::size() const {
    return layers_.size();
}

}
