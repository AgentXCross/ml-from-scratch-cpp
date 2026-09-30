# C++ Machine Learning Library from Scratch

A C++17 machine learning library built from scratch using the C++ Standard Library. `cpp_ml` implements classical machine learning algorithms, neural network components, preprocessing utilities, and core tensor operations. 

The library follows a PyTorch-inspired design, where training loops are written explicitly, and the math behind forward passes, gradients, and parameter updates stays visible.

`cpp_ml` is designed as an educational library for learning machine learning and deep learning fundamentals without hiding their implementations behind high-level abstractions, commonly seen in libraries like scikit-learn. The goal is to provide a simple interface for users while keeping the algorithm implementations readable.

<p align="center">
  <img src="repo_banner.png" width="90%">
</p>

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Option 1: Using Directly with add_subdirectory](#option-1-using-directly-with-add_subdirectory)
4. [Option 2: Using After Installing with find_package](#option-2-using-after-installing-with-find_package)
5. [Including Headers](#including-headers)
6. [Quick Start](#quick-start)
7. [Running Examples](#running-examples)
8. [Repo Structure](#repo-structure)
9. [C++ Code Conventions](#c-code-conventions)
10. [Implementation Status](#implementation-status)
11. [Current Status](#current-status)
12. [License](#license)

## Requirements

- C++17-compatible compiler (Clang or GCC)
- CMake 3.16+

## Installation 

Clone the repository either inside your project if you plan to use `add_subdirectory`, or anywhere on your machine if you plan to install the library and use `find_package`.

```bash
git clone https://github.com/AgentXCross/ml-from-scratch-cpp.git
```

Then, continue with Option 1 or 2 below.

## Option 1: Using Directly with add_subdirectory

This option is recommended if you want to include `cpp_ml` directly inside another CMake project.

Clone the repository somewhere inside your project. A common convention is to place third-party dependencies inside an external/ directory. For example:

```text
my_project/
├── CMakeLists.txt
├── main.cpp
└── external/
    └── ml-from-scratch-cpp/
```

Then, add the following to `CMakeLists.txt`:

```cmake 
cmake_minimum_required(VERSION 3.16)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/ml-from-scratch-cpp)

add_executable(my_app main.cpp)

target_link_libraries(my_app PRIVATE cpp_ml::cpp_ml)
```

Example `main.cpp`:

```cpp
#include <cpp_ml/cpp_ml.hpp>

int main() {
    cpp_ml::Tensor X{1.0};
    return 0;
}
```

Configure and build the project:

```bash
cmake -S . -B build
cmake --build build
```

With Option 1, `cpp_ml` is built as part of your project.

## Option 2: Using After Installing with find_package

This option allows `cpp_ml` to be installed once and reused by other CMake projects.

First, clone the repository and enter it:

```bash
git clone https://github.com/AgentXCross/ml-from-scratch-cpp.git
cd ml-from-scratch-cpp
```

Configure and build the library:

```bash
cmake -S . -B build
cmake --build build
```

Install it to a local directory:

```bash
cmake --install build --prefix install
```

This creates an installation containing the public headers, compiled library, and CMake package configuration:

```text
install/
├── include/
│   └── cpp_ml/
├── lib/
│   ├── libcpp_ml.a
│   └── cmake/
│       └── cpp_ml/
│           ├── cpp_mlConfig.cmake
│           └── cpp_mlTargets.cmake
```

Your separate project can then have a structure such as:

```bash
my_project/
├── CMakeLists.txt
└── main.cpp
```

Example `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cpp_ml CONFIG REQUIRED)

add_executable(my_app main.cpp)

target_link_libraries(my_app PRIVATE cpp_ml::cpp_ml)
```

When configuring `my_project`, tell CMake where `cpp_ml` was installed:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/ml-from-scratch-cpp/install
cmake --build build
```

## Including Headers 

Public headers are included through the `cpp_ml/` prefix:

```cpp
#include <cpp_ml/cpp_ml.hpp>
```

All library code lives in the `cpp_ml` namespace.

## Quick Start

```cpp
#include <cpp_ml/cpp_ml.hpp>

int main() {
    cpp_ml::Tensor X = cpp_ml::Tensor::from_flat_vector(
        {1.0, 2.0, 3.0},
        {3, 1}
    );

    cpp_ml::Tensor y = cpp_ml::Tensor::from_flat_vector(
        {2.0, 4.0, 6.0},
        {3, 1}
    );

    cpp_ml::LinearRegression model(1);

    model.fit(X, y, 100, 0.1);

    cpp_ml::Tensor predictions = model.predict(X);
}
```

Additional usage examples can be found in the `examples/` directory.

## Running Examples 

Examples are built through CMake targets:

```bash
cmake --build build --target logistic_regression_example
./build/logistic_regression_example
```

A helper script is also available to help run examples:

```bash 
./scripts/run_target.sh logistic_regression_example
```


## Repo Structure

```text
root/
├── CMakeLists.txt              Main CMake build file.
├── LICENSE                     MIT License.
├── README.md                   Documentation.
├── cmake/                      CMake package config template files.
├── data/                       Small datasets used by examples.
├── examples/                   Model usage examples.
├── include/                    Public header files.
│   └── cpp_ml/                 Public library include tree.
├── scripts/                    Python dataset prep and shell scripts.
├── src/                        Source implementations.
│   ├── core/                   Core building blocks.
│   │   ├── activations/        Activation functions and their gradients, also in layer form.
│   │   ├── layers/             Neural network layers.
│   │   ├── loss_functions/     Loss functions and their gradients.
│   │   ├── metrics/            Evaluation metrics.
│   │   ├── utils/              Helper operations.
│   │   └── tensor.cpp          Tensor class and operations.
│   ├── models/                 Model implementations.
│   ├── optim/                  Optimizers and parameter update logic (Not currently used).
│   └── preprocessing/          Data loading, splitting, and scaling.
└── tests/                      Correctness checks.
```

## C++ Code Conventions

1. Class names are written in PascalCase.
2. Functions, variables, parameters, and any other identifiers are written in snake_case.
3. Private data members in classes are written using trailing underscores (e.g. data_, size_).
4. The `using namespace std`; directive is never used. When referring to entities from the Standard Library, always use the `std` namespace followed by the scope resolution operator.
5. Brace initialization is preferred whenever possible (e.g. int x{5}).
6. Constructor methods prefer intializer list syntax whenever possible.
7. Header files all use `#pragma once`.
8. Empty tensors are not allowed, but scalar tensors are allowed.
9. Shape checks should happen before mathematical operations.
10. Vectors use parentheses initialization `std::vector<T>(size, value)`.
11. Runtime validation throws standard exceptions if an error occures. 

## Implementation Status

### Classical Machine Learning Algorithms

- [x] Linear Regression
- [x] Logistic Regression
- [ ] Ridge / Lasso / Elastic Net
- [x] Perceptron
- [x] ADALINE
- [x] Softmax Regression
- [x] K-Nearest Neighbors
- [ ] K-Nearest Neighbors Regressor
- [x] Gaussian Naive Bayes
- [ ] Categorical Naive Bayes
- [ ] Bernoulli Naive Bayes
- [ ] Mixed Naive Bayes
- [x] Decision Tree Classifier
- [ ] Decision Tree Regressor
- [x] Random Forest Classifier
- [ ] Random Forest Regressor
- [ ] Gradient Boosted Trees
- [ ] Linear SVM
- [ ] K-Means Clustering
- [ ] PCA

### Neural Network Components

#### Layers

- [x] Linear
- [ ] Conv1D
- [ ] Conv2D
- [ ] MaxPool1D
- [ ] MaxPool2D
- [ ] Flatten
- [ ] Dropout
- [ ] BatchNorm

#### Activations

- [x] Sigmoid
- [x] ReLU
- [x] Leaky ReLU
- [x] tanh
- [x] GeLU
- [ ] Softmax (Currently only implemented as a function)

### Loss Functions

- [x] Mean-Square Error (MSE)
- [x] Mean-Absolute Error (MAE)
- [x] Huber Loss
- [x] Binary Cross-Entropy (BCE)
- [x] Cross-Entropy

### Utilities

- [x] Sequential
- [ ] Weight Initialization
- [ ] Model Save / Load
- [ ] Train / Eval Mode

## Current Status

This project is still under development. The API may change as new models, neural network components, and performance improvements are added. Future work includes completing the checklist above, autograd implementation, and improving the performance of tensor operations through parallelism and multithreading.

## License

This project is licensed under the MIT License.
