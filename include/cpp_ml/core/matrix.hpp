#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace cpp_ml {

/*
Matrix class.

Depreciated. Replaced with the Tensor class.
*/
class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, double value);

    std::size_t rows() const;
    std::size_t cols() const;

    double &at(std::size_t row, std::size_t col); // read + write
    double at(std::size_t row, std::size_t col) const; // read-only

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double scalar) const;

    Matrix elementwise_multiply(const Matrix &other) const;

    Matrix matmul(const Matrix &other) const;
    Matrix transpose() const;

    std::pair<std::size_t, std::size_t> shape() const;

    // class methods
    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix ones(std::size_t rows, std::size_t cols);
    static Matrix random(std::size_t rows, std::size_t cols, double min = -1.0, double max = 1.0);

    void print() const;
    void print_shape() const;

    Matrix row(std::size_t row_index) const;

    static Matrix from_vector(
        const std::vector<std::vector<double>> &values
    );
};

}
