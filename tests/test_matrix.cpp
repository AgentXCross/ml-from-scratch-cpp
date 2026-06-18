#include "core/matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

bool close(double a, double b, double tolerance = 1e-9) {
    return std::abs(a - b) < tolerance;
}

void test_constructor_and_shape() {
    Matrix A(2, 3);

    assert(A.rows() == 2);
    assert(A.cols() == 3);

    auto shape = A.shape();

    assert(shape.first == 2);
    assert(shape.second == 3);
}

void test_default_values_are_zero() {
    Matrix A(2, 2);

    assert(close(A.at(0, 0), 0.0));
    assert(close(A.at(0, 1), 0.0));
    assert(close(A.at(1, 0), 0.0));
    assert(close(A.at(1, 1), 0.0));
}

void test_from_vector() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    assert(A.rows() == 2);
    assert(A.cols() == 2);

    assert(close(A.at(0, 0), 1.0));
    assert(close(A.at(0, 1), 2.0));
    assert(close(A.at(1, 0), 3.0));
    assert(close(A.at(1, 1), 4.0));
}

void test_zeros_and_ones() {
    Matrix Z = Matrix::zeros(2, 2);
    Matrix O = Matrix::ones(2, 2);

    assert(close(Z.at(0, 0), 0.0));
    assert(close(Z.at(1, 1), 0.0));

    assert(close(O.at(0, 0), 1.0));
    assert(close(O.at(1, 1), 1.0));
}

void test_addition() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Matrix B = Matrix::from_vector({
        {10.0, 20.0},
        {30.0, 40.0}
    });

    Matrix C = A + B;

    assert(close(C.at(0, 0), 11.0));
    assert(close(C.at(0, 1), 22.0));
    assert(close(C.at(1, 0), 33.0));
    assert(close(C.at(1, 1), 44.0));
}

void test_subtraction() {
    Matrix A = Matrix::from_vector({
        {10.0, 20.0},
        {30.0, 40.0}
    });

    Matrix B = Matrix::from_vector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Matrix C = A - B;

    assert(close(C.at(0, 0), 9.0));
    assert(close(C.at(0, 1), 18.0));
    assert(close(C.at(1, 0), 27.0));
    assert(close(C.at(1, 1), 36.0));
}

void test_scalar_multiply() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Matrix B = A * 2.0;

    assert(close(B.at(0, 0), 2.0));
    assert(close(B.at(0, 1), 4.0));
    assert(close(B.at(1, 0), 6.0));
    assert(close(B.at(1, 1), 8.0));
}

void test_elementwise_multiply() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Matrix B = Matrix::from_vector({
        {10.0, 20.0},
        {30.0, 40.0}
    });

    Matrix C = A.elementwise_multiply(B);

    assert(close(C.at(0, 0), 10.0));
    assert(close(C.at(0, 1), 40.0));
    assert(close(C.at(1, 0), 90.0));
    assert(close(C.at(1, 1), 160.0));
}

void test_matmul() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });

    Matrix B = Matrix::from_vector({
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    });

    Matrix C = A.matmul(B);

    assert(C.rows() == 2);
    assert(C.cols() == 2);

    assert(close(C.at(0, 0), 58.0));
    assert(close(C.at(0, 1), 64.0));
    assert(close(C.at(1, 0), 139.0));
    assert(close(C.at(1, 1), 154.0));
}

void test_transpose() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });

    Matrix T = A.transpose();

    assert(T.rows() == 3);
    assert(T.cols() == 2);

    assert(close(T.at(0, 0), 1.0));
    assert(close(T.at(0, 1), 4.0));
    assert(close(T.at(1, 0), 2.0));
    assert(close(T.at(1, 1), 5.0));
    assert(close(T.at(2, 0), 3.0));
    assert(close(T.at(2, 1), 6.0));
}

void test_out_of_range_access_throws() {
    Matrix A(2, 2);

    bool threw = false;

    try {
        A.at(2, 0);
    }
    catch (const std::out_of_range &) {
        threw = true;
    }

    assert(threw);
}

void test_shape_mismatch_addition_throws() {
    Matrix A(2, 2);
    Matrix B(3, 2);

    bool threw = false;

    try {
        Matrix C = A + B;
    }
    catch (const std::invalid_argument &) {
        threw = true;
    }

    assert(threw);
}

void test_row_extraction() {
    Matrix A = Matrix::from_vector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });

    Matrix row = A.row(1);

    assert(row.rows() == 1);
    assert(row.cols() == 3);

    assert(close(row.at(0, 0), 4.0));
    assert(close(row.at(0, 1), 5.0));
    assert(close(row.at(0, 2), 6.0));
}

int main() {
    test_constructor_and_shape();
    test_default_values_are_zero();
    test_from_vector();
    test_zeros_and_ones();
    test_addition();
    test_subtraction();
    test_scalar_multiply();
    test_elementwise_multiply();
    test_matmul();
    test_transpose();
    test_out_of_range_access_throws();
    test_shape_mismatch_addition_throws();
    test_row_extraction();

    std::cout << "All Matrix tests passed.\n";

    return 0;
}