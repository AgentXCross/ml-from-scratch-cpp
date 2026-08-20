#include "models/knn_regressor.hpp"

#include "core/utils/euclidean_distance.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

KNNRegressor::KNNRegressor() 
    : X_train_(Tensor()),
      y_train_(Tensor()),
      k_(1),
      fitted_(false) {}


static int validate_k(int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    return k;
}


KNNRegressor::KNNRegressor(int k) 
    : X_train_(Tensor()),
      y_train_(Tensor()),
      k_(validate_k(k)),
      fitted_(false) {}


void KNNRegressor::fit(
    const Tensor &X_train, 
    const Tensor &y_train
) {
    if (!X_train.is_matrix() || !y_train.is_matrix()) {
        throw std::invalid_argument("X_train and y_train must be rank-2 tensors");
    }

    if (X_train.rows() == 0 || X_train.cols() == 0) {
        throw std::invalid_argument("X_train cannot be empty");
    }

    if (y_train.rows() == 0 || y_train.cols() == 0) {
        throw std::invalid_argument("y_train cannot be empty");
    }

    if (X_train.rows() != y_train.rows()) {
        throw std::invalid_argument("X_train and y_train must have the same number of rows");
    }

    if (y_train.cols() != 1) {
        throw std::invalid_argument("y_train must be a column vector");
    }

    if (k_ > X_train.rows()) {
        throw std::invalid_argument("k cannot be larger than the number of training samples");
    }

    X_train_ = X_train;
    y_train_ = y_train;
    fitted_ = true;
}


Tensor KNNRegressor::predict(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (!fitted_) {
        throw std::runtime_error("KNNRegressor must be fitted before calling predict");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of columns as X_train");
    }

    Tensor predictions({X.rows(), 1});

    for (int i = 0; i < X.rows(); i++) {
        std::vector<std::pair<double, double>> distances_and_values;

        Tensor sample = X.row(i);
        assert(sample.rows() == 1);
        assert(sample.cols() == X_train_.cols());

        for (int j = 0; j < X_train_.rows(); j++) {
            Tensor train_sample = X_train_.row(j);
            assert(train_sample.rows() == 1);
            assert(train_sample.cols() == sample.cols());

            double distance = euclidean_distance(sample, train_sample);
            double value = static_cast<double>(y_train_.at(j, 0));

            distances_and_values.push_back({distance, value});
        }

        std::sort(
            distances_and_values.begin(),
            distances_and_values.end(),
            [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                return a.first < b.first;
            }
        );
        
        assert(distances_and_values.size() >= k_);
        double sum_of_neighbors = 0.0;

        for (int neighbor = 0; neighbor < k_; neighbor++) {
            sum_of_neighbors += distances_and_values[neighbor].second;
        }

        double average_of_neighbors = static_cast<double>(sum_of_neighbors) / k_;

        predictions.at(i, 0) = average_of_neighbors;
    }

    assert(predictions.is_matrix());
    assert(predictions.cols() == 1);
    assert(predictions.rows() == X.rows());

    return predictions;
}


int KNNRegressor::k() const {
    return k_;
}