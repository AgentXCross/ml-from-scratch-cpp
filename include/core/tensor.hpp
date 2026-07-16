#pragma once

#include <string>
#include <vector>

class Tensor {
private:
    std::vector<double> data_;
    std::vector<int> shape_;

    // Returns flattened index
    int compute_flat_index(const std::vector<int> &indices) const;

public:
    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(
        const std::vector<int> &shape,
        double fill_value
    );

    // static functions belong to the class itself, not an instance

    // Makes a 2D Tensor / Matrix from a vector of vector of doubles
    static Tensor from_vector(const std::vector<std::vector<double>> &values);

    // Returns a tensor of 0's
    static Tensor zeros(const std::vector<int>);

    // Returns a tensor of 1's
    static Tensor ones(const std::vector<int>);

    // Returns a tensor with random values within a range
    static Tensor random(
        const std::vector<int> &shape,
        double min_value = -0.01,
        double max_value = 0.01
    );

    int ndim() const;

    std::vector<int> shape() const;

    int size() const;

    int rows() const; // Only for rank 2

    int cols() const; // Only for rank 2

    double &at(const std::vector<int> &indices);

    double at(const std::vector<int> &indices) const;

    double &at(int row, int col); // Only for rank 2

    double at(int row, int col) const; // Only for rank 2

    Tensor row(int row_index) const;

    Tensor transpose() const;

    Tensor matmul(const Tensor &other) const;

    Tensor elementwise_multiply(const Tensor &other) const;

    Tensor operator+(const Tensor &other) const;

    Tensor operator-(const Tensor &other) const;

    Tensor operator*(double scalar) const;

    void print() const;

    void print_shape() const;
};