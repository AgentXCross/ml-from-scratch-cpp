#include "core/tensor.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

Tensor::Tensor() {
    data_ = std::vector<double>();
    shape_ = std::vector<int>();
}


Tensor::Tensor(const std::vector<int> &shape) {
    if (shape.empty()) {
        throw std::invalid_argument("shape cannot be empty");
    }

    int total_size = 1;
    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
        if (shape[i] <= 0) {
            throw std::invalid_argument("All shape dimensions must be positive");
        }

        total_size *= shape[i];
    }

    shape_ = shape;
    data_ = std::vector<double>(total_size, 0.0);
}


Tensor::Tensor(
    const std::vector<int> &shape,
    double fill_value
) {
    if (shape.empty()) {
        throw std::invalid_argument("shape cannot be empty");
    }

    int total_size = 1;

    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
        if (shape[i] <= 0) {
            throw std::invalid_argument("All shape dimensions must be positive");
        }

        total_size *= shape[i];
    }

    shape_ = shape;
    data_ = std::vector<double>(total_size, fill_value);
}


int Tensor::compute_flat_index(const std::vector<int> &indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    int flat_index = 0;
    int stride = 1; // the last dimension always has stride 1, stride is the number of steps
                    // needed to move in the flat array when you increase one index by 1 in a particular dimension

    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) { // start from the last dimension
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }

        flat_index += indices[i] * stride;
        stride *= shape_[i]; // compute the stride of the previous dimension 
    }

    return flat_index;
}


Tensor Tensor::from_vector(const std::vector<std::vector<double>> &values) {
    if (values.empty()) {
        return Tensor();
    }

    int num_rows = static_cast<int>(values.size());
    int num_cols = static_cast<int>(values[0].size());

    if (num_cols == 0) {
        return Tensor();
    }

    Tensor result({num_rows, num_cols});

    for (int i = 0; i < num_rows; i++) {
        if (static_cast<int>(values[i].size()) != num_cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }

        for (int j = 0; j < num_cols; j++) {
            result.at(i, j) = values[i][j];
        }
    }

    return result;
}


Tensor Tensor::zeros(const std::vector<int> &shape) {
    return Tensor(shape, 0.0);
}


Tensor Tensor::ones(const std::vector<int> &shape) {
    return Tensor(shape, 1.0);
}


Tensor Tensor::random(
    const std::vector<int> &shape,
    double min_value,
    double max_value
) {
    if (min_value > max_value) {
        throw std::invalid_argument("min_value must be less than max_value");
    }

    Tensor result(shape);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(min_value, max_value);

    for (int i = 0; i < result.size(); i++) {
        result.data_[i] = distribution(generator);
    }

    return result;
}


int Tensor::ndim() const {
    return static_cast<int>(shape_.size());
}


std::vector<int> Tensor::shape() const {
    return shape_;
}


int Tensor::size() const {
    return static_cast<int>(data_.size());
}


int Tensor::rows() const {
    if (ndim() != 2) {
        throw std::runtime_error("rows() only works for matrices/tensors of rank 2");
    }

    return shape_[0];
}


int Tensor::cols() const {
    if (ndim() != 2) {
        throw std::runtime_error("cols() only works for matrices/tensors of rank 2");
    }

    return shape_[1];
}


double &Tensor::at(const std::vector<int> &indices) {
    int flat_index = compute_flat_index(indices);

    return data_[flat_index];
}


double Tensor::at(const std::vector<int> &indices) const {
    int flat_index = compute_flat_index(indices);

    return data_[flat_index];
}


double &Tensor::at(int row, int col) {
    return at({row, col});
}


double Tensor::at(int row, int col) const {
    return at({row, col});
}


Tensor Tensor::row(int row_index) const {
    if (ndim() != 2) {
        throw std::runtime_error("row() only works for matrices/tensors of rank 2");
    }

    if (row_index < 0 || row_index >= rows()) {
        throw std::out_of_range("row_index is out of range");
    }

    Tensor result({1, cols()});

    for (int j = 0; j < cols(); j++) {
        result.at(0, j) = at(row_index, j);
    }

    return result;
}


Tensor Tensor::squeeze() const {
    if (ndim() == 0) {
        throw std::runtime_error("Cannot squeeze an empty tensor");
    }

    std::vector<int> new_shape;

    for (int i = 0; i < ndim(); i++) {
        if (shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::squeeze(int axis) const {
    if (ndim() == 0) {
        throw std::runtime_error("Cannot squeeze an empty tensor");
    }

    if (axis < 0 || axis >= ndim()) {
        throw std::out_of_range("axis is out of range");
    }

    if (shape_[axis] != 1) {
        throw std::invalid_argument("Cannot squeeze an axis whose size is not 1");
    }

    std::vector<int> new_shape;

    for (int i = 0; i < ndim(); i++) {
        if (i == axis) {
            continue;
        }

        new_shape.push_back(shape_[i]);
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::unsqueeze(int axis) const {
    if (size() == 0) {
        throw std::runtime_error("Cannot unsqueeze an empty tensor");
    }

    if (axis < 0 || axis >= ndim()) {
        throw std::out_of_range("axis is out of range");
    }

    std::vector<int> new_shape;

    for (int i = 0; i < axis; i++) {
        new_shape.push_back(shape_[i]);
    }

    new_shape.push_back(1);

    for (int i = axis; i < ndim(); i++) {
        new_shape.push_back(shape_[i]);
    }

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::permute(const std::vector<int> &axes) const {
    if (size() == 0) {
        throw std::runtime_error("Cannot permute an empty tensor");
    }

    if (static_cast<int>(axes.size()) != ndim()) {
        throw std::invalid_argument("axes size must match the number of tensor dimensions");
    }

    // Check that axes contains all ints from 0 to ndim - 1
    std::vector<bool> seen(ndim(), false);

    for (int i = 0; i < ndim(); i++) {
        if (axes[i] < 0 || axes[i] >= ndim()) {
            throw std::out_of_range("axis is out of range");
        }

        if (seen[axes[i]]) {
            throw std::invalid_argument("axes cannot contain duplicates");
        }

        seen[axes[i]] = true;
    }

    std::vector<int> new_shape;

    for (int i = 0; i < ndim(); i++) {
        new_shape.push_back(shape_[axes[i]]);
    }

    Tensor result(new_shape);

    for (int flat_index = 0; flat_index < size(); flat_index++) { // visit every element
        int remaining = flat_index;
        std::vector<int> old_indices(ndim());

        for (int i = ndim() - 1; i >= 0; i--) {
            old_indice
        }
    }
}


Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::runtime_error("transpose() only works on matrices/tensors of rank 2");
    }

    Tensor result({cols(), rows()});

    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}


Tensor Tensor::matmul(const Tensor &other) const {
    if (ndim() != 2) {
        throw std::runtime_error("matmul() only works on matrices/tensors of rank 2");
    }

    if (other.ndim() != 2) {
        throw std::runtime_error("other must be a matrix/tensor of rank 2");
    }

    if (cols() != other.rows()) {
        throw std::invalid_argument("Left tensor cols must match rows of right/other tensor");
    }

    Tensor result({rows(), other.cols()});

    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            double sum = 0.0;

            for (int k = 0; k < cols(); k++) {
                sum += at(i, k) * other.at(k, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}


Tensor Tensor::elementwise_multiply(const Tensor &other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Both tensors must have the same shape");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
}


Tensor Tensor::operator+(const Tensor &other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Both tensors must have the same shape");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}


Tensor Tensor::operator-(const Tensor &other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Both tensors must have the same shape");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}


Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}


void Tensor::print() const {
    if (ndim() ==  2) {
        for (int i = 0; i < rows(); i++) {
            std::cout << "[";

            for (int j = 0; j < cols(); j++) {
                std::cout << " " << at(i, j) << " ";
            }

            std::cout << "]" << "\n";
        }

        return;
    }

    for (int i = 0; i < size(); i++) {
        std::cout << data_[i] << " ";
    }

    std::cout << "\n";
}


void Tensor::print_shape() const {
    std::cout << "(";

    for (int i = 0; i < ndim(); i++) {
        std::cout << shape_[i];

        if (i != ndim() - 1) {
            std::cout << ", ";
        }
    }

    std::cout << ")" << "\n";
}