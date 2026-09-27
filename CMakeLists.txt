cmake_minimum_required(VERSION 3.16) # cmake version requirement

project(cpp_ml_from_scratch VERSION 0.1.0 LANGUAGES CXX) # project name, version, and language

set(CMAKE_CXX_STANDARD 17) # C++ version
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Regeneration command
# cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++ 

# Library
add_library(cpp_ml
    src/core/matrix.cpp
    src/core/tensor.cpp
    src/core/activations/relu.cpp
    src/core/activations/sigmoid.cpp
    src/core/activations/leaky_relu.cpp
    src/core/activations/tanh.cpp
    src/core/activations/gelu.cpp
    src/core/activations/softmax.cpp
    src/core/layers/linear.cpp
    src/core/layers/sequential.cpp
    src/core/loss_functions/mse.cpp
    src/core/loss_functions/mae.cpp
    src/core/loss_functions/binary_cross_entropy.cpp
    src/core/loss_functions/cross_entropy.cpp
    src/core/metrics/accuracy.cpp
    src/core/metrics/r2_score.cpp
    src/core/metrics/f1_score.cpp
    src/core/metrics/precision.cpp
    src/core/metrics/recall.cpp
    src/core/metrics/confusion_matrix.cpp
    src/core/utils/threshold.cpp
    src/core/utils/euclidean_distance.cpp
    src/core/utils/tree_utils.cpp
    src/models/linear_regression.cpp
    src/models/logistic_regression.cpp
    src/models/perceptron.cpp
    src/models/adaline.cpp
    src/models/knn.cpp
    src/models/knn_regressor.cpp
    src/models/softmax_regression.cpp
    src/models/gaussian_naive_bayes.cpp
    src/models/decision_tree_classifier.cpp
    src/models/decision_tree_regressor.cpp
    src/models/random_forest_classifier.cpp
    src/preprocessing/standard_scaler.cpp
    src/preprocessing/dataset.cpp
    src/preprocessing/min_max_scaler.cpp
    src/preprocessing/train_test_split.cpp
    src/preprocessing/one_hot_encode.cpp
)
add_library(cpp_ml::cpp_ml ALIAS cpp_ml)
target_include_directories(cpp_ml PUBLIC
    include
)

# ====== Tests ======

# Matrix 
# cmake --build build --target test_matrix
add_executable(test_matrix
    tests/test_matrix.cpp
)
target_link_libraries(test_matrix PRIVATE
    cpp_ml
)


# ====== Examples ======

# Linear regression example
# cmake --build build --target linear_regression_example
add_executable(linear_regression_example
    examples/linear_regression_example.cpp
)
# Link to library
target_link_libraries(linear_regression_example PRIVATE
    cpp_ml
)

# Logistic regression example
# cmake --build build --target logistic_regression_example
add_executable(logistic_regression_example
    examples/logistic_regression_example.cpp
)
target_link_libraries(logistic_regression_example PRIVATE
    cpp_ml
)

# Perceptron example
# cmake --build build --target perceptron_example
add_executable(perceptron_example
    examples/perceptron_example.cpp
)
target_link_libraries(perceptron_example
    cpp_ml
)

# ADALINE example
# cmake --build build --target adaline_example
add_executable(adaline_example
    examples/adaline_example.cpp
)
target_link_libraries(adaline_example
    cpp_ml
)

# KNN example
# cmake --build build --target knn_example
add_executable(knn_example
    examples/knn_example.cpp
)
target_link_libraries(knn_example
    cpp_ml
)

# Softmax regression example
# cmake --build build --target softmax_regression_example
add_executable(softmax_regression_example
    examples/softmax_regression_example.cpp
)
target_link_libraries(softmax_regression_example
    cpp_ml
)

# Random forest example
# cmake --build build --target random_forest_example
add_executable(random_forest_example
    examples/random_forest_example.cpp
)
target_link_libraries(random_forest_example
    cpp_ml
)

# MLP example
# cmake --build build --target mlp_example
add_executable(mlp_example
    examples/mlp_example.cpp
)
target_link_libraries(mlp_example
    cpp_ml
)
