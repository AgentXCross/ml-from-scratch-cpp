#include "models/perceptron.hpp"

#include "core/utils/threshold.hpp"

#include <cassert>
#include <stdexcept>

Perceptron::Perceptron() 
    : weights_(Tensor()),
      bias_(Tensor()) {}


static int validate_num_features(int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be positive");
    }

    return num_features;
}


Perceptron::Perceptron(int num_features) 
    : weights_(Tensor::random({validate_num_features(num_features), 1})),
      bias_(Tensor::random({1, 1})) {}   


Tensor Perceptron::predict(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    Tensor scores = X.matmul(weights_);
    assert(scores.is_matrix());
    assert(scores.rows() == X.rows() && scores.cols() == 1);
    
    for (int i = 0; i < scores.rows(); i++) {
        scores.at(i, 0) = scores.at(i, 0) + bias_.at(0, 0);
    }

    return threshold(scores, 0.0);
}


void Perceptron::train_epoch(
    const Tensor &X,
    const Tensor &y_true,
    double learning_rate
) {
    // Make predictions on X
    // Error for each sample is difference between y_true and the prediction
    // Weights are adjusted every sample if the error is not exactly 0
    // When adjusting, add the lr * error * feature_value
    if (!X.is_matrix() || !y_true.is_matrix()) {
        throw std::invalid_argument("X and y_true must both be rank-2 tensors");
    }

    if (X.rows() != y_true.rows()) {
        throw std::invalid_argument("X and y_true must have the same number of rows");
    }

    if (y_true.cols() != 1) {
        throw std::invalid_argument("y_true must have exactly one column");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weights");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    Tensor predictions = predict(X);
    assert(predictions.is_matrix());
    assert(predictions.rows() == X.rows() && predictions.cols() == 1);

    for (int i = 0; i < X.rows(); i++) {
        double error = y_true.at(i, 0) - predictions.at(i, 0);

        if (error != 0.0) {
            for (int j = 0; j < X.cols(); j++) {
                weights_.at(j, 0) += learning_rate * error * X.at(i, j);
            }

            bias_.at(0, 0) += learning_rate * error;
        }
    }
}


Tensor Perceptron::weights() const {
    return weights_;
}


Tensor Perceptron::bias() const {
    return bias_;
}