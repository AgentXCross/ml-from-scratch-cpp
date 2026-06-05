#include "core/matrix.hpp"
#include <stdexcept>
#include <random>

Matrix::Matrix() {
    rows_ = 0;
    cols_ = 0;
}

Matrix::Matrix(int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions cannot be negative");
    }
    rows_ = rows;
    cols_ = cols;
    data_ = std::vector<double>(rows * cols, 0.0);
}