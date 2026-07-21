#pragma once

#include <vector>


/*
Tensor class.

A tensor is an n-dimensional array of double values. Tensors can have any
number of dimensions (rank >= 0).

Empty tensors are not allowed. Empty tensors are tensors with size = 0.
Scalar tensors are allowed. Scalar tensors are rank-0 tensors with size = 1 and shape = {}.
*/
class Tensor {
private:
    std::vector<double> data_;
    std::vector<int> shape_;

    // Returns flattened from indices
    int indices_to_flat(const std::vector<int> &indices) const;

    // Returns indices from flatted
    std::vector<int> flat_to_indices(int flat) const;

public:
    Tensor(); // Creates a scalar tensor with element 0.0
    Tensor(double scalar); // Creates a scalar tensor (size = 1, ndim = 0)
    Tensor(const std::vector<int> &shape);
    Tensor(
        const std::vector<int> &shape,
        double fill_value
    );

    // static functions belong to the class itself, not an instance

    // Makes a 2D Tensor / Matrix from a vector of vector of doubles
    static Tensor from_vector(const std::vector<std::vector<double>> &values);

    // Returns a tensor of 0's
    static Tensor zeros(const std::vector<int> &shape);

    // Returns a tensor of 1's
    static Tensor ones(const std::vector<int> &shape);

    // Returns a tensor with random values within a range
    static Tensor random(
        const std::vector<int> &shape,
        double min_value = -0.01,
        double max_value = 0.01
    );

    int ndim() const;
    const std::vector<int> &shape() const;

    int size() const;
    int rows() const; // Only for rank 2
    int cols() const; // Only for rank 2

    double &at(const std::vector<int> &indices);
    double at(const std::vector<int> &indices) const;
    double &at(int row, int col); // Only for rank 2
    double at(int row, int col) const; // Only for rank 2
    double &at_flat(int flat);
    double at_flat(int flat) const;

    Tensor row(int row_index) const; // Only for rank 2
    Tensor col(int col_index) const; // Only for rank 2

    Tensor squeeze() const; // Removes all dimensions of size 1
    Tensor squeeze(int axis) const; // Removes the specified dimension if its size is 1
    Tensor unsqueeze(int axis) const; // Inserts a new dimension of size 1 at the specified axis
    Tensor permute(const std::vector<int> &axes) const; // Reorder the tensor's dimensions, axes contains 0 to ndim - 1
    Tensor reshape(const std::vector<int> &new_shape) const; // Change the tensor's shape while preserving the data order
    Tensor transpose() const; // Only for rank 2
    Tensor transpose(int axis_1, int axis_2) const; // Transpose generalized to higher dimensions, swaps the 2 axes
    Tensor matmul(const Tensor &other) const; // Only for rank 2 (for now)
    Tensor elementwise_multiply(const Tensor &other) const;

    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor &operator+=(const Tensor &other);
    Tensor &operator-=(const Tensor &other);
    Tensor &operator+=(double scalar);
    Tensor &operator-=(double scalar);
    Tensor &operator*=(double scalar);
    Tensor &operator/=(double scalar);
    bool operator==(const Tensor &other) const; // compares shape and values
    bool operator!=(const Tensor &other) const; // compares shape and values

    void print() const; // Prints a matrix for rank 2, flat otherwise
    void print_shape() const;

    bool is_scalar() const; // ndim = 0, size = 1
    bool is_matrix() const; // ndim = 2
    bool empty() const; // size = 0
    bool has_same_shape(const Tensor &other) const; // compares shape

    double sum() const;
    double mean() const;
    double min() const;
    double max() const;
    int argmin() const; // flat argmin index
    int argmax() const; // flat argmax index
    std::vector<int> argmin_indices() const;
    std::vector<int> argmax_indices() const;

    void fill(double scalar); // fill an existing Tensor
    Tensor flatten() const;
    Tensor clone() const;
    Tensor contiguous() const;

    Tensor operator-() const; // negation operator
    Tensor abs() const;
    Tensor square() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(double exponent) const;
};
