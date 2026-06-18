#include "core/utils/threshold.hpp"

Matrix threshold(
    const Matrix &x, 
    double cutoff,
    double upper,
    double lower
) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            if (x.at(i, j) >= cutoff) {
                result.at(i, j) = upper;
            } else {
                result.at(i, j) = lower;
            }
        }
    }

    return result;
}