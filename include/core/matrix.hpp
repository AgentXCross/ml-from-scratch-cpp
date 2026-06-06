// matrix.hpp

# pragma once

#include <iostream>
#include <vector>

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

    // Matmul and T
    Matrix matmul(const Matrix &other) const;
    Matrix transpose() const;

    // static means that this function belongs to this class itself, not individual objects
    static Matrix random(int rows, int cols, double min = -1.0, double max = 1.0);

    // print the matrix
    void print() const;
};