#pragma once

#include "core/matrix.hpp"

Matrix relu(const Matrix &x);

Matrix relu_gradient(const Matrix &x);