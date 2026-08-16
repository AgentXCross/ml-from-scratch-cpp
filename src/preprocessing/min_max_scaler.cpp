#include "preprocessing/min_max_scaler.hpp"

#include <cassert>
#include <stdexcept>

MinMaxScaler::MinMaxScaler()
    : mins_(Tensor()),
      maxs_(Tensor()),
      fitted_(false) {}


void MinMaxScaler::fit(const Tensor &X) {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot fit MinMaxScaler on an empty tensor");
    }

    mins_ = Tensor({1, X.cols()});
    maxs_ = Tensor({1, X.cols()});

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

    assert(mins_.is_matrix());
    assert(maxs_.is_matrix());
    assert(mins_.rows() == 1 && mins_.cols() == X.cols());
    assert(maxs_.rows() == 1 && maxs_.cols() == X.cols());

    fitted_ = true;
}


Tensor MinMaxScaler::transform(const Tensor &X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before calling transform");
    }

    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("Cannot transform an empty tensor");
    }

    if (X.cols() != mins_.cols()) {
        throw std::invalid_argument("X must have the same number of columns as the fitted data");
    }

    Tensor result({X.rows(), X.cols()});

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            if (maxs_.at(0, j) - mins_.at(0, j) == 0.0) {
                result.at(i, j) = 0.0;
                continue;
            }
            result.at(i, j) = (X.at(i, j) - mins_.at(0, j)) / (maxs_.at(0, j) - mins_.at(0, j));
        }
    }

    assert(result.is_matrix());
    assert(result.rows() == X.rows());
    assert(result.cols() == X.cols());

    return result;
}


Tensor MinMaxScaler::fit_transform(const Tensor &X) {
    fit(X);
    
    return transform(X);
}


Tensor MinMaxScaler::mins() const {
    return mins_;
}


Tensor MinMaxScaler::maxs() const {
    return maxs_;
}
