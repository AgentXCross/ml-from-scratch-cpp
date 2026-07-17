#include "core/tensor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

Tensor::Tensor() {
    shape_ = std::vector<int>();
    data_ = std::vector<double>{0.0};
}


Tensor::Tensor(double scalar) {
    data_ = std::vector<double>{scalar};
    shape_ = std::vector<int>();
}


Tensor::Tensor(const std::vector<int> &shape) {
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


int Tensor::indices_to_flat(const std::vector<int> &indices) const {
    if (empty()) {
        throw std::runtime_error("Cannot index an empty tensor");
    }

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


std::vector<int> Tensor::flat_to_indices(int flat) const {
    if (empty()) {
        throw std::runtime_error("Cannot convert flat index for an empty tensor");
    }

    if (flat < 0 || flat >= size()) {
        throw std::out_of_range("flat is out of range");
    }

    std::vector<int> indices(ndim());

    for (int i = ndim() - 1; i >= 0; i--) {
        indices[i] = flat % shape_[i];
        flat /= shape_[i];
    }

    return indices;
}


Tensor Tensor::from_vector(const std::vector<std::vector<double>> &values) {
    if (values.empty()) {
        throw std::invalid_argument("values cannot be empty");
    }

    int num_rows = static_cast<int>(values.size());
    int num_cols = static_cast<int>(values[0].size());

    if (num_cols == 0) {
        throw std::invalid_argument("values cannot have empty rows");
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


const std::vector<int> &Tensor::shape() const {
    return shape_;
}


int Tensor::size() const {
    return static_cast<int>(data_.size());
}


int Tensor::rows() const {
    if (!is_matrix()) {
        throw std::runtime_error("rows() only works for matrices/tensors of rank 2");
    }

    return shape_[0];
}


int Tensor::cols() const {
    if (!is_matrix()) {
        throw std::runtime_error("cols() only works for matrices/tensors of rank 2");
    }

    return shape_[1];
}


double &Tensor::at(const std::vector<int> &indices) {
    int flat_index = indices_to_flat(indices);

    return data_[flat_index];
}


double Tensor::at(const std::vector<int> &indices) const {
    int flat_index = indices_to_flat(indices);

    return data_[flat_index];
}


double &Tensor::at(int row, int col) {
    return at({row, col});
}


double Tensor::at(int row, int col) const {
    return at({row, col});
}


Tensor Tensor::row(int row_index) const {
    if (!is_matrix()) {
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


Tensor Tensor::col(int col_index) const {
    if (!is_matrix()) {
        throw std::runtime_error("col() only works for matrices/tensors of rank 2");
    }

    if (col_index < 0 || col_index >= cols()) {
        throw std::out_of_range("Column index out of range");
    }

    Tensor result({rows(), 1});

    for (int i = 0; i < rows(); i++) {
        result.at(i, 0) = at(i, col_index);
    }

    return result;
}


Tensor Tensor::squeeze() const {
    if (empty()) {
        throw std::runtime_error("Cannot squeeze an empty tensor");
    }

    if (is_scalar()) {
        return *this;
    }

    std::vector<int> new_shape;

    for (int i = 0; i < ndim(); i++) {
        if (shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
        }
    }

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::squeeze(int axis) const {
    if (empty()) {
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

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::unsqueeze(int axis) const {
    if (empty()) {
        throw std::runtime_error("Cannot unsqueeze an empty tensor");
    }

    if (axis < 0 || axis > ndim()) {
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
    if (empty()) {
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

        if (seen[axes[i]] == true) { // already seen this axis
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
        std::vector<int> old_indices = flat_to_indices(flat_index);
        
        std::vector<int> new_indices(ndim());

        for (int i = 0; i < ndim(); i++) {
            new_indices[i] = old_indices[axes[i]];
        }

        result.at(new_indices) = data_[flat_index];
    }

    return result;
}


Tensor Tensor::reshape(const std::vector<int> &new_shape) const {
    if (empty()) {
        throw std::runtime_error("Cannot reshape an empty tensor");
    }

    int new_size = 1;

    for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
        if (new_shape[i] <= 0) {
            throw std::invalid_argument("All shape dimensions must be positive");
        }

        new_size *= new_shape[i];
    }

    if (new_size != size()) {
        throw std::invalid_argument("new_shape must have the same total size as tensor");
    }

    Tensor result(new_shape);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::transpose() const {
    if (!is_matrix()) {
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


Tensor Tensor::transpose(int axis_1, int axis_2) const {
    if (empty()) {
        throw std::runtime_error("Cannot transpose an empty tensor");
    }

    if (axis_1 < 0 || axis_1 >= ndim()) {
        throw std::out_of_range("axis_1 is out of range");
    }

    if (axis_2 < 0 || axis_2 >= ndim()) {
        throw std::out_of_range("axis_2 is out of range");
    }

    std::vector<int> axes;

    for (int i = 0; i < ndim(); i++) {
        axes.push_back(i);
    }

    int temp = axes[axis_1];
    axes[axis_1] = axes[axis_2];
    axes[axis_2] = temp;

    return permute(axes);
}


Tensor Tensor::matmul(const Tensor &other) const {
    if (!is_matrix()) {
        throw std::runtime_error("matmul() only works on matrices/tensors of rank 2");
    }

    if (!other.is_matrix()) {
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
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot multiply empty tensors");
    }

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
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot add empty tensors");
    }

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
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot subtract empty tensors");
    }

    if (shape_ != other.shape_) {
        throw std::invalid_argument("Both tensors must have the same shape");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}


Tensor Tensor::operator+(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot add to an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] + scalar;
    }

    return result;
}


Tensor Tensor::operator-(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot subtract from an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] - scalar;
    }

    return result;
}


Tensor Tensor::operator*(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot multiply an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}


Tensor Tensor::operator/(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot divide an empty tensor");
    }

    if (scalar == 0.0) {
        throw std::invalid_argument("Cannot divide by 0");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] / scalar;
    }

    return result;
}


Tensor &Tensor::operator+=(const Tensor &other) {
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot add empty tensors");
    }

    if (!has_same_shape(other)) {
        throw std::invalid_argument("Tensor shapes must match");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] += other.data_[i];
    }

    return *this;
}


Tensor &Tensor::operator-=(const Tensor &other) {
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot subtract empty tensors");
    }

    if (!has_same_shape(other)) {
        throw std::invalid_argument("Tensor shapes must match");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] -= other.data_[i];
    }

    return *this;
}


Tensor &Tensor::operator+=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot add to an empty tensor");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] += scalar;
    }

    return *this;
}


Tensor &Tensor::operator-=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot subtract from an empty tensor");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] -= scalar;
    }

    return *this;
}


Tensor &Tensor::operator*=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot multiply an empty tensor");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] *= scalar;
    }

    return *this;
}


Tensor &Tensor::operator/=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot divide an empty tensor");
    }

    if (scalar == 0.0) {
        throw std::invalid_argument("Cannot divide by 0");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] /= scalar;
    }

    return *this;
}


bool Tensor::operator==(const Tensor &other) const {
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot compare empty tensors");
    }

    if (shape_ != other.shape_) {
        return false;
    }

    for (int i = 0; i < size(); i++) {
        if (data_[i] != other.data_[i]) {
            return false;
        }
    }

    return true;
}


bool Tensor::operator!=(const Tensor &other) const {
    return !(*this == other);
}


void Tensor::print() const {
    if (empty()) {
        throw std::runtime_error("Cannot print an empty tensor");
    }

    if (is_matrix()) {
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


bool Tensor::is_scalar() const {
    if (ndim() == 0 && size() == 1) {
        return true;
    }

    return false;
}


bool Tensor::is_matrix() const {
    return (ndim() == 2);
}


bool Tensor::empty() const {
    return (size() == 0);
}


bool Tensor::has_same_shape(const Tensor &other) const {
    return (shape_ == other.shape_);
}


double Tensor::sum() const {
    if (empty()) {
        throw std::runtime_error("Cannot sum an empty tensor");
    }

    double sum = 0.0;

    for (int i = 0; i < size(); i++) {
        sum += data_[i];
    }

    return sum;
}


double Tensor::mean() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate mean of an empty tensor");
    } 

    double sum = 0.0;

    for (int i = 0; i < size(); i++) {
        sum += data_[i];
    }

    return sum / size();
}


double Tensor::min() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate min of an empty tensor");
    }

    double cur_min = std::numeric_limits<double>::infinity();

    for (int i = 0; i < size(); i++) {
        if (data_[i] < cur_min) {
            cur_min = data_[i];
        }
    }

    return cur_min;
}


double Tensor::max() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate max of an empty tensor");
    }

    double cur_max = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < size(); i++) {
        if (data_[i] > cur_max) {
            cur_max = data_[i];
        }
    }

    return cur_max;
}


int Tensor::argmin() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate argmin of an empty tensor");
    }

    double cur_min = std::numeric_limits<double>::infinity();
    int min_idx = 0;

    for (int i = 0; i < size(); i++) {
        if (data_[i] < cur_min) {
            cur_min = data_[i];
            min_idx = i;
        }
    }

    return min_idx;
}


int Tensor::argmax() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate argmax of an empty tensor");
    }

    double cur_max = -std::numeric_limits<double>::infinity();
    int max_idx = 0;

    for (int i = 0; i < size(); i++) {
        if (data_[i] > cur_max) {
            cur_max = data_[i];
            max_idx = i;
        }
    }

    return max_idx;
}


std::vector<int> Tensor::argmin_indices() const {
    int flat_min_idx = argmin();

    return flat_to_indices(flat_min_idx);
}


std::vector<int> Tensor::argmax_indices() const {
    int flat_max_idx = argmax();

    return flat_to_indices(flat_max_idx);
}


void Tensor::fill(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot fill an empty tensor");
    }

    for (int i = 0; i < size(); i++) {
        data_[i] = scalar;
    }

    return;
}


Tensor Tensor::flatten() const {
    if (empty()) {
        throw std::runtime_error("Cannot flaten an empty tensor");
    }

    Tensor result({1, size()});

    for (int i = 0; i < size(); i++) {
        result.at(0, i) = data_[i];
    }

    return result;
}


Tensor Tensor::clone() const {
    if (empty()) {
        throw std::runtime_error("Cannot clone an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::contiguous() const {
    return *this;
}


Tensor Tensor::operator-() const {
    if (empty()) {
        throw std::invalid_argument("Cannot negate an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = -data_[i];
    }

    return result;
}


Tensor Tensor::abs() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the absolute value of an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = std::abs(data_[i]);
    }

    return result;
}


Tensor Tensor::square() const {
    if (empty()) {
        throw std::invalid_argument("Cannot square an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = data_[i] * data_[i];
    }

    return result;
}


Tensor Tensor::sqrt() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the square root of an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        if (data_[i] < 0.0) {
            throw std::invalid_argument("Cannot take square root of negative values");
        }

        result.data_[i] = std::sqrt(data_[i]);
    }

    return result;
}


Tensor Tensor::exp() const {
    if (empty()) {
        throw std::invalid_argument("Cannot exponentiate an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = std::exp(data_[i]);
    }

    return result;
}


Tensor Tensor::log() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the natural log of an empty tensor");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        if (data_[i] <= 0.0) {
            throw std::invalid_argument("Cannot take log of non-positive values");
        }

        result.data_[i] = std::log(data_[i]);
    }

    return result;
}


Tensor Tensor::pow(double exponent) const {
    if (empty()) {
        throw std::invalid_argument("Cannot raise an empty tensor to a power");
    }

    Tensor result(shape_);

    for (int i = 0; i < size(); i++) {
        result.data_[i] = std::pow(data_[i], exponent);
    }

    return result;
}