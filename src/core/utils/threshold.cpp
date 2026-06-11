#include "core/utils/threshold.hpp"

Matrix threshold(const Matrix &x, double cutoff) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) >= cutoff) {
                result.at(i, j) = 1.0;
            } else {
                result.at(i, j) = 0.0;
            }
        }
    }

    return result;
}