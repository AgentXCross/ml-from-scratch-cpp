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
    data_ = std::vector<double> (rows * cols, 0.0);
}

Matrix::Matrix(int rows, int cols, double value) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions cannot be negative");
    }
    rows_ = rows;
    cols_ = cols;
    data_ = std::vector<double> (rows * cols, value);
}

int Matrix::rows() const {
    return rows_;
}

int Matrix::cols() const {
    return cols_;
}

double &Matrix::at(int row, int col) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}

double Matrix::at(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_[row * cols_ + col];
}

Matrix Matrix::operator+(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows_, cols_);

    for (int i = 0; i < rows_ * cols_; i++) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows_, cols_);

    for (int i = 0; i < rows_ * cols_; i++) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);

    for (int i = 0; i < rows_ * cols_; i++) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

Matrix Matrix::matmul(const Matrix &other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }

    Matrix result(rows_, other.cols_);

    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < other.cols_; j++) {
            double sum = 0.0;

            for (int k = 0; k < cols_; k++) {
                sum += at(i, k) * other.at(k, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
    Matrix result(rows, cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for (int i = 0; i < rows * cols; i++) {
        result.data_[i] = dist(gen);
    }

    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            std::cout << at(i, j) << " ";
        }
        std::cout << "\n";
    }
}