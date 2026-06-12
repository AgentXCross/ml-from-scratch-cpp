// matrix.hpp

# pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <utility>

class Matrix {
private:
    int rows_;
    int cols_;
    std::vector<double> data_;

public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);

    // Getters
    int rows() const;
    int cols() const;

    // Element access
    double &at(int row, int col); // read + write
    double at(int row, int col) const; // read-only

    // Operator override
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double scalar) const;

    // Element-wise multiplication
    Matrix elementwise_multiply(const Matrix &other) const;

    // Matmul and T
    Matrix matmul(const Matrix &other) const;
    Matrix transpose() const;

    // Shape
    std::pair<int, int> shape() const;

    // static means that this function belongs to this class itself, not individual objects
    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix random(int rows, int cols, double min = -1.0, double max = 1.0);

    // print the matrix
    void print() const;
    void print_shape() const;

    // Initialize Matrix from a vector of vectors
    static Matrix from_vector(
        const std::vector<std::vector<double>> &values
    );
};
