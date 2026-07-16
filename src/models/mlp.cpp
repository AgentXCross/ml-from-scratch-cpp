#include "models/mlp.hpp"

#include "core/activations/relu.hpp"
#include "core/activations/sigmoid.hpp"
#include "core/layers/linear.hpp"

#include <memory>
#include <stdexcept>

MLP::MLP() {
    network_ = Sequential();
}

MLP::MLP(const std::vector<int> &layer_sizes) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("layer_sizes must contain at least input size and output size");
    }

    for (int i = 0; i < layer_sizes.size(); i++) {
        if (layer_sizes[i] <= 0) {
            throw std::invalid_argument("all layer sizes must be positive");
        }
    }

    network_ = Sequential();

    
}