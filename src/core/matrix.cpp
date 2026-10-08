#include "cpp_ml/core/matrix.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

namespace cpp_ml {

Matrix::Matrix() 
    : rows_{0},
      cols_{0},
      data_{} {}


Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_{rows},
      cols_{cols},
      data_{std::vector<double>(rows_ * cols_, 0.0)} {}


Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_{rows},
      cols_{cols},
      data_{std::vector<double>(rows_ * cols_, value)} {}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

double &Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}

double Matrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}

Matrix Matrix::operator+(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result{rows_, cols_};

    for (std::size_t i{0}; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result{rows_, cols_};

    for (std::size_t i{0}; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result{rows_, cols_};

    for (std::size_t i{0}; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

Matrix Matrix::elementwise_multiply(const Matrix &other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }

    Matrix result{rows_, cols_};

    for (std::size_t i{0}; i < rows_; ++i) {
        for (std::size_t j{0}; j < cols_; ++j) {
            result.at(i, j) = at(i, j) * other.at(i, j);
        }
    }

    return result;
}

Matrix Matrix::matmul(const Matrix &other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }

    Matrix result{rows_, other.cols_};

    for (std::size_t i{0}; i < rows_; ++i) {
        for (std::size_t j{0}; j < other.cols_; ++j) {
            double sum{0.0};

            for (std::size_t k{0}; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result{cols_, rows_};

    for (std::size_t i{0}; i < rows_; ++i) {
        for (std::size_t j{0}; j < cols_; ++j) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}

std::pair<std::size_t, std::size_t> Matrix::shape() const {
    return std::pair<std::size_t, std::size_t>{rows_, cols_};
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix{rows, cols, 0.0};
}

Matrix Matrix::ones(std::size_t rows, std::size_t cols) {
    return Matrix{rows, cols, 1.0};
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, double min, double max) {
    Matrix result{rows, cols};

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<double> dist{min, max};

    for (std::size_t i{0}; i < rows * cols; ++i) {
        result.data_[i] = dist(gen);
    }

    return result;
}

void Matrix::print() const {
    for (std::size_t i{0}; i < rows_; ++i) {
        for (std::size_t j{0}; j < cols_; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::print_shape() const {
    std::cout << "(" << rows_ << ", " << cols_ << ")\n";
}

Matrix Matrix::row(std::size_t row_index) const {
    if (row_index >= rows_) {
        throw std::out_of_range("Matrix row index is out of range");
    }

    Matrix result{1, cols_};

    for (std::size_t j{0}; j < cols_; ++j) {
        result.at(0, j) = at(row_index, j);
    }

    return result;
}

Matrix Matrix::from_vector(
    const std::vector<std::vector<double>> &values
) {
    std::size_t rows{values.size()};

    if (rows == 0) {
        return Matrix{};
    }

    std::size_t cols{values[0].size()};

    Matrix result{rows, cols};

    for (std::size_t i{0}; i < rows; ++i) {
        if (values[i].size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }

        for (std::size_t j{0}; j < cols; ++j) {
            result.at(i, j) = values[i][j];
        }
    }

    return result;
}

}
