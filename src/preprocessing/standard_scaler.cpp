#include "preprocessing/standard_scaler.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

StandardScaler::StandardScaler()
    : means_(Tensor()),
      stds_(Tensor()),
      fitted_(false) {}


void StandardScaler::fit(const Tensor &X) {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot fit StandardScaler on an empty tensor");
    }

    means_ = Tensor({1, X.cols()});
    stds_ = Tensor({1, X.cols()});

    for (int j = 0; j < X.cols(); j++) { // loop through features
        double sum = 0.0;

        for (int i = 0; i < X.rows(); i++) { // loop through samples
            sum = sum + X.at(i, j);
        }

        double mean = sum / X.rows();
        means_.at(0, j) = mean;

        double squared_diff_sum = 0.0;

        for (int i = 0; i < X.rows(); i++) {
            double diff = X.at(i, j) - mean;
            squared_diff_sum = squared_diff_sum + diff * diff;
        }

        double variance = squared_diff_sum / X.rows();
        double std = std::sqrt(variance);

        if (std == 0.0) {
            std = 1.0;
        }

        stds_.at(0, j) = std;
    }

    assert(means_.is_matrix());
    assert(stds_.is_matrix());
    assert(means_.rows() == 1 && means_.cols() == X.cols());
    assert(stds_.rows() == 1 && stds_.cols() == X.cols());

    fitted_ = true;
}


Tensor StandardScaler::transform(const Tensor &X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before calling transform");
    }

    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot transform an empty tensor");
    }

    if (X.cols() != means_.cols()) {
        throw std::invalid_argument("X must have same number of columns as the fitted data");
    }

    Tensor result({X.rows(), X.cols()});

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            result.at(i, j) = (X.at(i, j) - means_.at(0, j)) / stds_.at(0, j);
        }
    }

    assert(result.is_matrix());
    assert(result.rows() == X.rows());
    assert(result.cols() == X.cols());

    return result;
}


Tensor StandardScaler::fit_transform(const Tensor &X) {
    fit(X);

    return transform(X);
}


Tensor StandardScaler::means() const {
    return means_;
}


Tensor StandardScaler::stds() const {
    return stds_;
}
