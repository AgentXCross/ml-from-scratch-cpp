#include "cpp_ml/core/tensor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace cpp_ml {

Tensor::Tensor() 
    : data_{0.0},
      shape_{} {}


Tensor::Tensor(double scalar) 
    : data_{scalar},
      shape_{} {}


static std::size_t compute_total_size(const std::vector<std::size_t> &shape) {
    std::size_t total_size{1};

    for (std::size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("All shape dimensions must be positive");
        }

        total_size *= dim;
    }

    return total_size;
}


Tensor::Tensor(const std::vector<std::size_t> &shape)
    : data_{std::vector<double>(compute_total_size(shape), 0.0)},
      shape_{shape} {}


Tensor::Tensor(
    const std::vector<std::size_t> &shape,
    double fill_value
)
    : data_{std::vector<double>(compute_total_size(shape), fill_value)},
      shape_{shape} {}


std::size_t Tensor::indices_to_flat(const std::vector<std::size_t> &indices) const {
    if (empty()) {
        throw std::runtime_error("Cannot index an empty tensor");
    }

    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    std::size_t flat_index{0};
    std::size_t stride{1}; // the last dimension always has stride 1, stride is the number of steps
                    // needed to move in the flat array when you increase one index by 1 in a particular dimension

    for (std::size_t i{shape_.size()}; i-- > 0;) { // start from the last dimension
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }

        flat_index += indices[i] * stride;
        stride *= shape_[i]; // compute the stride of the previous dimension 
    }

    return flat_index;
}


std::vector<std::size_t> Tensor::flat_to_indices(std::size_t flat) const {
    if (empty()) {
        throw std::runtime_error("Cannot convert flat index for an empty tensor");
    }

    if (flat >= size()) {
        throw std::out_of_range("flat is out of range");
    }

    std::vector<std::size_t> indices(ndim());

    for (std::size_t i{ndim()}; i-- > 0;) {
        indices[i] = flat % shape_[i];
        flat /= shape_[i];
    }

    return indices;
}


Tensor Tensor::from_vector(const std::vector<std::vector<double>> &values) {
    if (values.empty()) {
        throw std::invalid_argument("values cannot be empty");
    }

    std::size_t num_rows{values.size()};
    std::size_t num_cols{values[0].size()};

    if (num_cols == 0) {
        throw std::invalid_argument("values cannot have empty rows");
    }

    Tensor result{{num_rows, num_cols}};

    for (std::size_t i{0}; i < num_rows; ++i) {
        if (values[i].size() != num_cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }

        for (std::size_t j{0}; j < num_cols; ++j) {
            result.at(i, j) = values[i][j];
        }
    }

    return result;
}


Tensor Tensor::from_flat_vector(
    const std::vector<double> &values,
    const std::vector<std::size_t> &shape
) {
    if (values.empty()) {
        throw std::invalid_argument("values cannot be empty");
    }

    if (shape.empty()) {
        if (values.size() != 1) {
            throw std::invalid_argument("Scalar tensors must have exactly one value");
        }
        
        return Tensor{values[0]};
    }

    std::size_t total_size{compute_total_size(shape)};

    if (values.size() != total_size) {
        throw std::invalid_argument("Number of values does not match the tensor shape");
    }

    Tensor tensor{shape};

    for (std::size_t i{0}; i < total_size; ++i) {
        tensor.at_flat(i) = values[i];
    }

    return tensor;
}


Tensor Tensor::zeros(const std::vector<std::size_t> &shape) {
    return Tensor{shape, 0.0};
}


Tensor Tensor::ones(const std::vector<std::size_t> &shape) {
    return Tensor{shape, 1.0};
}


Tensor Tensor::random(
    const std::vector<std::size_t> &shape,
    double min_value,
    double max_value
) {
    if (min_value > max_value) {
        throw std::invalid_argument("min_value must be less than max_value");
    }

    Tensor result{shape};

    std::random_device rd{};
    std::mt19937 generator{rd()};
    std::uniform_real_distribution<double> distribution{min_value, max_value};

    for (std::size_t i{0}; i < result.size(); ++i) {
        result.data_[i] = distribution(generator);
    }

    return result;
}


std::size_t Tensor::ndim() const {
    return shape_.size();
}


const std::vector<std::size_t> &Tensor::shape() const {
    return shape_;
}


std::size_t Tensor::size() const {
    return data_.size();
}


std::size_t Tensor::rows() const {
    if (!is_matrix()) {
        throw std::runtime_error("rows() only works for matrices/tensors of rank 2");
    }

    return shape_[0];
}


std::size_t Tensor::cols() const {
    if (!is_matrix()) {
        throw std::runtime_error("cols() only works for matrices/tensors of rank 2");
    }

    return shape_[1];
}


double &Tensor::at(const std::vector<std::size_t> &indices) {
    std::size_t flat_index{indices_to_flat(indices)};

    return data_[flat_index];
}


double Tensor::at(const std::vector<std::size_t> &indices) const {
    std::size_t flat_index{indices_to_flat(indices)};

    return data_[flat_index];
}


double &Tensor::at(std::size_t row, std::size_t col) {
    return at({row, col});
}


double Tensor::at(std::size_t row, std::size_t col) const {
    return at({row, col});
}


double &Tensor::at_flat(std::size_t flat) {
    if (empty()) {
        throw std::runtime_error("Cannot index an empty tensor");
    }

    if (flat >= size()) {
        throw std::out_of_range("flat is out of range");
    }

    return data_[flat];
}


double Tensor::at_flat(std::size_t flat) const {
    if (empty()) {
        throw std::runtime_error("Cannot index an empty tensor");
    }

    if (flat >= size()) {
        throw std::out_of_range("flat is out of range");
    }

    return data_[flat];
}


Tensor Tensor::row(std::size_t row_index) const {
    if (!is_matrix()) {
        throw std::runtime_error("row() only works for matrices/tensors of rank 2");
    }

    if (row_index >= rows()) {
        throw std::out_of_range("row_index is out of range");
    }

    Tensor result{{1, cols()}};

    for (std::size_t j{0}; j < cols(); ++j) {
        result.at(0, j) = at(row_index, j);
    }

    return result;
}


Tensor Tensor::col(std::size_t col_index) const {
    if (!is_matrix()) {
        throw std::runtime_error("col() only works for matrices/tensors of rank 2");
    }

    if (col_index >= cols()) {
        throw std::out_of_range("Column index out of range");
    }

    Tensor result{{rows(), 1}};

    for (std::size_t i{0}; i < rows(); ++i) {
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

    std::vector<std::size_t> new_shape;

    for (std::size_t i{0}; i < ndim(); ++i) {
        if (shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
        }
    }

    Tensor result{new_shape};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::squeeze(std::size_t axis) const {
    if (empty()) {
        throw std::runtime_error("Cannot squeeze an empty tensor");
    }

    if (axis >= ndim()) {
        throw std::out_of_range("axis is out of range");
    }

    if (shape_[axis] != 1) {
        throw std::invalid_argument("Cannot squeeze an axis whose size is not 1");
    }

    std::vector<std::size_t> new_shape;

    for (std::size_t i{0}; i < ndim(); ++i) {
        if (i == axis) {
            continue;
        }

        new_shape.push_back(shape_[i]);
    }

    Tensor result{new_shape};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::unsqueeze(std::size_t axis) const {
    if (empty()) {
        throw std::runtime_error("Cannot unsqueeze an empty tensor");
    }

    if (axis > ndim()) {
        throw std::out_of_range("axis is out of range");
    }

    std::vector<std::size_t> new_shape;

    for (std::size_t i{0}; i < axis; ++i) {
        new_shape.push_back(shape_[i]);
    }

    new_shape.push_back(1);

    for (std::size_t i{axis}; i < ndim(); ++i) {
        new_shape.push_back(shape_[i]);
    }

    Tensor result{new_shape};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::permute(const std::vector<std::size_t> &axes) const {
    if (empty()) {
        throw std::runtime_error("Cannot permute an empty tensor");
    }

    if (axes.size() != ndim()) {
        throw std::invalid_argument("axes size must match the number of tensor dimensions");
    }

    // Check that axes contains all ints from 0 to ndim - 1
    std::vector<bool> seen(ndim(), false);

    for (std::size_t i{0}; i < ndim(); ++i) {
        if (axes[i] >= ndim()) {
            throw std::out_of_range("axis is out of range");
        }

        if (seen[axes[i]] == true) { // already seen this axis
            throw std::invalid_argument("axes cannot contain duplicates");
        }

        seen[axes[i]] = true;
    }

    std::vector<std::size_t> new_shape;

    for (std::size_t i{0}; i < ndim(); ++i) {
        new_shape.push_back(shape_[axes[i]]);
    }

    Tensor result{new_shape};

    for (std::size_t flat_index{0}; flat_index < size(); ++flat_index) { // visit every element
        std::vector<std::size_t> old_indices{flat_to_indices(flat_index)};
        
        std::vector<std::size_t> new_indices(ndim());

        for (std::size_t i{0}; i < ndim(); ++i) {
            new_indices[i] = old_indices[axes[i]];
        }

        result.at(new_indices) = data_[flat_index];
    }

    return result;
}


Tensor Tensor::reshape(const std::vector<std::size_t> &new_shape) const {
    if (empty()) {
        throw std::runtime_error("Cannot reshape an empty tensor");
    }

    std::size_t new_size{1};

    for (std::size_t i{0}; i < new_shape.size(); ++i) {
        if (new_shape[i] == 0) {
            throw std::invalid_argument("All shape dimensions must be positive");
        }

        new_size *= new_shape[i];
    }

    if (new_size != size()) {
        throw std::invalid_argument("new_shape must have the same total size as tensor");
    }

    Tensor result{new_shape};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i];
    }

    return result;
}


Tensor Tensor::transpose() const {
    if (!is_matrix()) {
        throw std::runtime_error("transpose() only works on matrices/tensors of rank 2");
    }

    Tensor result{{cols(), rows()}};

    for (std::size_t i{0}; i < rows(); ++i) {
        for (std::size_t j{0}; j < cols(); ++j) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}


Tensor Tensor::transpose(std::size_t axis_1, std::size_t axis_2) const {
    if (empty()) {
        throw std::runtime_error("Cannot transpose an empty tensor");
    }

    if (axis_1 >= ndim()) {
        throw std::out_of_range("axis_1 is out of range");
    }

    if (axis_2 >= ndim()) {
        throw std::out_of_range("axis_2 is out of range");
    }

    std::vector<std::size_t> axes;

    for (std::size_t i{0}; i < ndim(); ++i) {
        axes.push_back(i);
    }

    std::size_t temp{axes[axis_1]};
    axes[axis_1] = axes[axis_2];
    axes[axis_2] = temp;

    return permute(axes);
}


Tensor Tensor::batched_matmul(
    const Tensor &a,
    const Tensor &b
) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Neither tensor can be empty");
    }

    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("batched_matmul requires tensors be at least rank-3");
    }

    std::size_t m{a.shape_[a.ndim() - 2]};
    std::size_t k{a.shape_[a.ndim() - 1]};

    std::size_t b_k{b.shape_[b.ndim() - 2]};
    std::size_t n{b.shape_[b.ndim() - 1]};

    if (k != b_k) {
        throw std::invalid_argument("Inner matrix dimensions must match");
    }

    std::size_t batch_dims{a.ndim() - 2}; // Number of dims in a excluding the last 2
    std::size_t b_batch_dims{b.ndim() - 2}; // Number of dims in b excluding the last 2

    if (batch_dims != b_batch_dims) {
        throw std::invalid_argument("Batch dimensions must match exactly");
    }

    std::vector<std::size_t> result_shape;

    for (std::size_t i{0}; i < batch_dims; ++i) {
        if (a.shape_[i] != b.shape_[i]) {
            throw std::invalid_argument("Batch dimensions must match exactly");
        }

        result_shape.push_back(a.shape_[i]);
    }

    result_shape.push_back(m);
    result_shape.push_back(n);

    Tensor result{result_shape};

    std::size_t batch_size{1};

    for (std::size_t i{0}; i < batch_dims; ++i) {
        batch_size *= a.shape_[i];
    }

    for (std::size_t batch_flat{0}; batch_flat < batch_size; ++batch_flat) {
        // convert the flat batch index into true indices
        std::vector<std::size_t> batch_indices(batch_dims);

        std::size_t remaining{batch_flat};

        for (std::size_t i{batch_dims}; i-- > 0;) {
            batch_indices[i] = remaining % a.shape_[i];
            remaining = remaining / a.shape_[i];
        }

        for (std::size_t row{0}; row < m; ++row) {
            for (std::size_t col{0}; col < n; ++col) {
                double sum{0.0};

                for (std::size_t inner{0}; inner < k; ++inner) {
                    std::vector<std::size_t> a_indices{batch_indices};
                    a_indices.push_back(row);
                    a_indices.push_back(inner);

                    std::vector<std::size_t> b_indices{batch_indices};
                    b_indices.push_back(inner);
                    b_indices.push_back(col);

                    sum += a.at(a_indices) * b.at(b_indices);
                } 

                std::vector<std::size_t> result_indices{batch_indices};
                result_indices.push_back(row);
                result_indices.push_back(col);

                result.at(result_indices) = sum;
            }
        }
    }

    return result;
}


Tensor Tensor::matmul(const Tensor &other) const {
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot compute matmul of empty tensors");
    }

    if (is_matrix() && other.is_matrix()) {
        if (cols() != other.rows()) {
            throw std::invalid_argument("Left tensor cols must match right tensor rows");
        }
        Tensor result{{rows(), other.cols()}};

        for (std::size_t i{0}; i < result.rows(); ++i) {
            for (std::size_t j{0}; j < result.cols(); ++j) {
                double sum{0.0};

                for (std::size_t k{0}; k < cols(); ++k) {
                    sum += at(i, k) * other.at(k, j);
                }

                result.at(i, j) = sum;
            }
        }

        return result;
    }

    return Tensor::batched_matmul(*this, other);
}


Tensor Tensor::elementwise_multiply(const Tensor &other) const {
    if (empty() || other.empty()) {
        throw std::runtime_error("Cannot multiply empty tensors");
    }

    if (shape_ != other.shape_) {
        throw std::invalid_argument("Both tensors must have the same shape");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}


Tensor Tensor::operator+(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot add to an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }

    return result;
}


Tensor Tensor::operator-(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot subtract from an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }

    return result;
}


Tensor Tensor::operator*(double scalar) const {
    if (empty()) {
        throw std::runtime_error("Cannot multiply an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    for (std::size_t i{0}; i < size(); ++i) {
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

    for (std::size_t i{0}; i < size(); ++i) {
        data_[i] -= other.data_[i];
    }

    return *this;
}


Tensor &Tensor::operator+=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot add to an empty tensor");
    }

    for (std::size_t i{0}; i < size(); ++i) {
        data_[i] += scalar;
    }

    return *this;
}


Tensor &Tensor::operator-=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot subtract from an empty tensor");
    }

    for (std::size_t i{0}; i < size(); ++i) {
        data_[i] -= scalar;
    }

    return *this;
}


Tensor &Tensor::operator*=(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot multiply an empty tensor");
    }

    for (std::size_t i{0}; i < size(); ++i) {
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

    for (std::size_t i{0}; i < size(); ++i) {
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

    for (std::size_t i{0}; i < size(); ++i) {
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
        for (std::size_t i{0}; i < rows(); ++i) {
            std::cout << "[";

            for (std::size_t j{0}; j < cols(); ++j) {
                std::cout << " " << at(i, j) << " ";
            }

            std::cout << "]" << "\n";
        }

        return;
    }

    for (std::size_t i{0}; i < size(); ++i) {
        std::cout << data_[i] << " ";
    }

    std::cout << "\n";
}


void Tensor::print_shape() const {
    std::cout << "(";

    for (std::size_t i{0}; i < ndim(); ++i) {
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

    double sum{0.0};

    for (std::size_t i{0}; i < size(); ++i) {
        sum += data_[i];
    }

    return sum;
}


double Tensor::mean() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate mean of an empty tensor");
    } 

    double sum{0.0};

    for (std::size_t i{0}; i < size(); ++i) {
        sum += data_[i];
    }

    return sum / static_cast<double>(size());
}


double Tensor::min() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate min of an empty tensor");
    }

    double cur_min{std::numeric_limits<double>::infinity()};

    for (std::size_t i{0}; i < size(); ++i) {
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

    double cur_max{-std::numeric_limits<double>::infinity()};

    for (std::size_t i{0}; i < size(); ++i) {
        if (data_[i] > cur_max) {
            cur_max = data_[i];
        }
    }

    return cur_max;
}


std::size_t Tensor::argmin() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate argmin of an empty tensor");
    }

    double cur_min{std::numeric_limits<double>::infinity()};
    std::size_t min_idx{0};

    for (std::size_t i{0}; i < size(); ++i) {
        if (data_[i] < cur_min) {
            cur_min = data_[i];
            min_idx = i;
        }
    }

    return min_idx;
}


std::size_t Tensor::argmax() const {
    if (empty()) {
        throw std::runtime_error("Cannot calculate argmax of an empty tensor");
    }

    double cur_max{-std::numeric_limits<double>::infinity()};
    std::size_t max_idx{0};

    for (std::size_t i{0}; i < size(); ++i) {
        if (data_[i] > cur_max) {
            cur_max = data_[i];
            max_idx = i;
        }
    }

    return max_idx;
}


std::vector<std::size_t> Tensor::argmin_indices() const {
    std::size_t flat_min_idx{argmin()};

    return flat_to_indices(flat_min_idx);
}


std::vector<std::size_t> Tensor::argmax_indices() const {
    std::size_t flat_max_idx{argmax()};

    return flat_to_indices(flat_max_idx);
}


void Tensor::fill(double scalar) {
    if (empty()) {
        throw std::runtime_error("Cannot fill an empty tensor");
    }

    for (std::size_t i{0}; i < size(); ++i) {
        data_[i] = scalar;
    }

    return;
}


Tensor Tensor::flatten() const {
    if (empty()) {
        throw std::runtime_error("Cannot flaten an empty tensor");
    }

    Tensor result{{1, size()}};

    for (std::size_t i{0}; i < size(); ++i) {
        result.at(0, i) = data_[i];
    }

    return result;
}


Tensor Tensor::clone() const {
    if (empty()) {
        throw std::runtime_error("Cannot clone an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = -data_[i];
    }

    return result;
}


Tensor Tensor::abs() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the absolute value of an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = std::abs(data_[i]);
    }

    return result;
}


Tensor Tensor::square() const {
    if (empty()) {
        throw std::invalid_argument("Cannot square an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = data_[i] * data_[i];
    }

    return result;
}


Tensor Tensor::sqrt() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the square root of an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = std::exp(data_[i]);
    }

    return result;
}


Tensor Tensor::log() const {
    if (empty()) {
        throw std::invalid_argument("Cannot take the natural log of an empty tensor");
    }

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
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

    Tensor result{shape_};

    for (std::size_t i{0}; i < size(); ++i) {
        result.data_[i] = std::pow(data_[i], exponent);
    }

    return result;
}

}
