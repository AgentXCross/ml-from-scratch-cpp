#pragma once

#include "core/matrix.hpp"

Matrix leaky_relu(const Matrix &x, double alpha);

Matrix leaky_relu_gradient(const Matrix &x, double alpha);