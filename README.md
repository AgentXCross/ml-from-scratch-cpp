# C++ Machine Learning Library from Scratch

A C++17 machine learning library built from scratch using the C++ Standard Library. `cpp_ml` implements classical machine learning algorithms, neural network components, preprocessing utilities, and core tensor operations. 

The library follows a PyTorch-inspired design, where training loops are written explicitly, and the math behind forward passes, gradients, and parameter updates stays visible.

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
9. [Implementation Status](#implementation-status)
10. [Current Status](#current-status)
11. [License](#license)

## Requirements

- C++17-compatible compiler (Clang or GCC)
- CMake 3.16+

## Installation 

Clone the repository from the top-level of your project (recommended for Option 1) or anywhere on your machine (if choosing Option 2):

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

This produces an installation containing the public headers, compiled library, and CMake package configuration:

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

## Option 1: Using Directly with add_subdirectory

If the repo is cloned inside another CMake project, it can be added directly.

Assuming the cloned `ml-from-scratch-cpp` folder is inside an `external` folder at the top level of the project:

```text
my_project/
├── CMakeLists.txt
├── main.cpp
└── external/
    └── ml-from-scratch-cpp/
```

Add the following to `CMakeLists.txt`:

```cmake 
add_subdirectory(external/ml-from-scratch-cpp)

add_executable(my_app main.cpp)

target_link_libraries(my_app PRIVATE cpp_ml::cpp_ml)
```

## Option 2: Using After Installing with find_package

This option is for using `cpp_ml` after it has already been built and installed somewhere on your local machine (not necessarily your projects folder).

Follow the instructions above from the `Installation` section. This should produce a local `install` folder, containing `include` and `lib`.

Now your CMake project can use the installed library. 

Example project:
```text
my_project/
├── CMakeLists.txt
└── main.cpp
```

`main.cpp`:
```cpp
#include <cpp_ml/cpp_ml.hpp>

int main() {
    cpp_ml::Tensor X(1.0);
    return 0;
}
```

`CMakeLists.txt`:
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
    cpp_ml::Tensor X = cpp_ml::Tensor::from_vector({
      {1.0},
      {2.0},
      {3.0}
    });

    cpp_ml::Tensor y = cpp_ml::Tensor::from_vector({
      {2.0},
      {4.0},
      {6.0}
    });

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
- [ ] Huber Loss
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