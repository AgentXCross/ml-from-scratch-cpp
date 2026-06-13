#include "preprocessing/standard_scaler.hpp"

#include <cmath>
#include <stdexcept>

StandardScaler::StandardScaler() {
    means_ = Matrix();
    stds_ = Matrix();
    fitted_ = false;
}

void StandardScaler::fit(const Matrix &X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot fit StandardScaler on an empty matrix");
    }

    means_ = Matrix(1, X.cols());
    stds_ = Matrix(1, X.cols());

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

    fitted_ = true;
}

Matrix StandardScaler::transform(const Matrix &X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before calling transform");
    }

    if (X.cols() != means_.cols()) {
        throw std::invalid_argument("X must have same number of columns as the fitted data");
    }

    Matrix result(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            result.at(i, j) = (X.at(i, j) - means_.at(0, j)) / stds_.at(0, j);
        }
    }

    return result;
}

Matrix StandardScaler::fit_transform(const Matrix &X) {
    fit(X);

    return transform(X);
}

Matrix StandardScaler::means() const {
    return means_;
}

Matrix StandardScaler::stds() const {
    return stds_;
}