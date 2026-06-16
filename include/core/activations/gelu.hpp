#pragma once

#include "core/matrix.hpp"

Matrix gelu(const Matrix &x);

Matrix gelu_gradient(const Matrix &x);