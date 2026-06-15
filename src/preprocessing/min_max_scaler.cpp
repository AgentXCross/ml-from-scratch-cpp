#include "preprocessing/min_max_scaler.hpp"

#include <stdexcept>

MinMaxScaler::MinMaxScaler() {
    mins_ = Matrix();
    maxs_ = Matrix();
    fitted_ = false;
}

void MinMaxScaler::fit(const Matrix &X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot fit MinMaxScaler on an empty matrix");
    }

    mins_ = Matrix(1, X.cols());
    maxs_ = Matrix(1, X.cols());

    for (int j = 0; j < X.cols(); j++) {
        double min_value = X.at(0, j);
        double max_value = X.at(0, j);

        for (int i = 0; i < X.rows(); i++) {
            double value = X.at(i, j);

            if (value < min_value) {
                min_value = value;
            }

            if (value > max_value) {
                max_value = value;
            }
        }

        mins_.at(0, j) = min_value;
        maxs_.at(0, j) = max_value;
    }

    fitted_ = true;
}

Matrix MinMaxScaler::transform(const Matrix &X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before calling transform");
    }

    if (X.cols() != mins_.cols()) {
        throw std::invalid_argument("X must have the same number of columns as the fitted data");
    }

    Matrix result(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            if (maxs_.at(0, j) - mins_.at(0, j) == 0.0) {
                result.at(i, j) = 0.0;
                continue;
            }
            result.at(i, j) = (X.at(i, j) - mins_.at(0, j)) / (maxs_.at(0, j) - mins_.at(0, j));
        }
    }

    return result;
}

Matrix MinMaxScaler::fit_transform(const Matrix &X) {
    fit(X);
    
    return transform(X);
}

Matrix MinMaxScaler::mins() const {
    return mins_;
}

Matrix MinMaxScaler::maxs() const {
    return maxs_;
}