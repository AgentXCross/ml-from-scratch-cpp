#include "models/softmax_regression.hpp"

#include "core/activations/softmax.hpp"

#include <cassert>
#include <stdexcept>

SoftmaxRegression::SoftmaxRegression()
    : weights_(Tensor()),
      bias_(Tensor()),
      dL_dw_(Tensor()),
      dL_db_(Tensor()) {}


static int validate_num_features(int num_features) {
    if (num_features <= 0) {
        throw std::invalid_argument("Number of features must be positive");
    }

    return num_features;
}


static int validate_num_classes(int num_classes) {
    if (num_classes < 2) {
        throw std::invalid_argument("SoftmaxRegression requires at least 2 classes");
    }

    return num_classes;
}


// weights are a rank-2 tensor {num_features, num_classes}
// Each column contains the weight vector for one class
SoftmaxRegression::SoftmaxRegression(int num_features, int num_classes) 
    : weights_(Tensor::random({validate_num_features(num_features), validate_num_classes(num_classes)}, -0.01, 0.01)),
      bias_(Tensor::random({1, num_classes}, -0.01, 0.01)),
      dL_dw_(Tensor({num_features, num_classes})),
      dL_db_(Tensor({1, num_classes})) {}


Tensor SoftmaxRegression::predict_probs(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weight rows");
    }

    Tensor logits = X.matmul(weights_);
    assert(logits.is_matrix());
    assert(logits.cols() == weights_.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < logits.cols(); j++) {
            logits.at(i, j) += bias_.at(0, j);
        }
    }

    return softmax(logits);
}

Tensor SoftmaxRegression::predict(const Tensor &X) const {
    Tensor probabilities = predict_probs(X);
    Tensor predictions({probabilities.rows(), 1});

    for (int i = 0; i < predictions.rows(); i++) {
        Tensor row = probabilities.row(i);
        assert(row.rows() == 1);
        int predicted_class = row.argmax();

        predictions.at(i, 0) = static_cast<double>(predicted_class);
    }

    return predictions;
}

void SoftmaxRegression::backward(
    const Tensor &X,
    const Tensor &dL_dlogits
) {
    // dL_dlogits comes from cross_entropy_gradient
    // logits = XW + b, where X is {num_samples, num_features}, W is {num_features, num_classes}
    // dL_dW = X.T @ dL_dlogits
    if (!X.is_matrix() || !dL_dlogits.is_matrix()) {
        throw std::invalid_argument("X and dL_dlogits must be rank-2 tensors");
    }

    if (X.rows() != dL_dlogits.rows()) {
        throw std::invalid_argument("X and dL_dlogits must have the same number of rows");
    }

    if (X.cols() != weights_.rows()) {
        throw std::invalid_argument("X columns must match number of weight rows");
    }

    if (dL_dlogits.cols() != weights_.cols()) {
        throw std::invalid_argument("dL_dlogits columns must match number of classes");
    }

    dL_dw_ = X.transpose().matmul(dL_dlogits);
    assert(dL_dw_.shape() == weights_.shape());

    for (int j = 0; j < dL_dlogits.cols(); j++) {
        double bias_gradient = 0.0;

        for (int i = 0; i < dL_dlogits.rows(); i++) {
            bias_gradient += dL_dlogits.at(i, j);
        }

        dL_db_.at(0, j) = bias_gradient;
    }

    assert(dL_db_.shape() == bias_.shape());
}

void SoftmaxRegression::step(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    weights_ = weights_ - (dL_dw_ * learning_rate);
    bias_ = bias_ - (dL_db_ * learning_rate);
}

Tensor SoftmaxRegression::weights() const {
    return weights_;
}

Tensor SoftmaxRegression::bias() const {
    return bias_;
}