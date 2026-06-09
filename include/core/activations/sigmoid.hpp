#pragma once

#include "core/matrix.hpp"

Matrix sigmoid(const Matrix &x);

Matrix sigmoid_gradient(const Matrix &x);