cmake_minimum_required(VERSION 3.16) # cmake version requirement

project(cpp_ml_from_scratch LANGUAGES CXX) # project name and language

set(CMAKE_CXX_STANDARD 17) # C++ version
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Regeneration command
# cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++ 

# Core
add_library(cpp_ml_core
    src/core/matrix.cpp
)
target_include_directories(cpp_ml_core PUBLIC
    include
)

# Preprocessing
add_library(cpp_ml_preprocessing
    src/preprocessing/standard_scaler.cpp
    src/preprocessing/dataset.cpp
    src/preprocessing/min_max_scaler.cpp
    src/preprocessing/train_test_split.cpp
)
target_link_libraries(cpp_ml_preprocessing PUBLIC
    cpp_ml_core
)

# Losses
add_library(cpp_ml_losses
    src/core/loss_functions/mse.cpp
    src/core/loss_functions/mae.cpp
    src/core/loss_functions/binary_cross_entropy.cpp
    src/core/loss_functions/cross_entropy.cpp
)
target_link_libraries(cpp_ml_losses PUBLIC
    cpp_ml_core
)

# Activations
add_library(cpp_ml_activations
    src/core/activations/relu.cpp
    src/core/activations/sigmoid.cpp
    src/core/activations/leaky_relu.cpp
    src/core/activations/tanh.cpp
    src/core/activations/gelu.cpp
    src/core/activations/softmax.cpp
)
target_link_libraries(cpp_ml_activations PUBLIC
    cpp_ml_core
)

# Metrics
add_library(cpp_ml_metrics
    src/core/metrics/accuracy.cpp
    src/core/metrics/r2_score.cpp
    src/core/metrics/f1_score.cpp
    src/core/metrics/precision.cpp
    src/core/metrics/recall.cpp
    src/core/metrics/confusion_matrix.cpp
)
target_link_libraries(cpp_ml_metrics PUBLIC
    cpp_ml_core
)

# Utils
add_library(cpp_ml_utils
    src/core/utils/threshold.cpp
    src/core/utils/argmax.cpp
    src/core/utils/argmin.cpp
    src/core/utils/euclidean_distance.cpp
)
target_link_libraries(cpp_ml_utils PUBLIC
    cpp_ml_core
)

# Models
add_library(cpp_ml_models
    src/models/linear_regression.cpp
    src/models/logistic_regression.cpp
    src/models/perceptron.cpp
    src/models/adaline.cpp
    src/models/knn.cpp
    src/models/softmax_regression.cpp
    src/models/naive_bayes.cpp
)
target_link_libraries(cpp_ml_models PUBLIC
    cpp_ml_core
    cpp_ml_activations
    cpp_ml_utils
)

# ====== Tests ======

# Matrix 
# cmake --build build --target test_matrix
add_executable(test_matrix
    tests/test_matrix.cpp
)
target_link_libraries(test_matrix PRIVATE
    cpp_ml_core;
)


# ====== Examples ======

# Linear regression example
# cmake --build build --target linear_regression_example
add_executable(linear_regression_example
    examples/linear_regression_example.cpp
)
# Link to library
target_link_libraries(linear_regression_example PRIVATE
    cpp_ml_models
    cpp_ml_losses
)

# Logistic regression example
# cmake --build build --target logistic_regression_example
add_executable(logistic_regression_example
    examples/logistic_regression_example.cpp
)
target_link_libraries(logistic_regression_example PRIVATE
    cpp_ml_models
    cpp_ml_losses
    cpp_ml_metrics
    cpp_ml_preprocessing
)