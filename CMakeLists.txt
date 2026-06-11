cmake_minimum_required(VERSION 3.16) # cmake version requirement

project(cpp_ml_from_scratch LANGUAGES CXX) # project name and language

set(CMAKE_CXX_STANDARD 17) # C++ version
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core
add_library(cpp_ml_core
    src/core/matrix.cpp
)
target_include_directories(cpp_ml_core PUBLIC
    include
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
)
target_link_libraries(cpp_ml_activations PUBLIC
    cpp_ml_core
)

# Metrics
add_library(cpp_ml_metrics
    src/core/metrics/accuracy.cpp
    src/core/metrics/r2_score.cpp
)
target_link_libraries(cpp_ml_metrics PUBLIC
    cpp_ml_core
)

# Models
add_library(cpp_ml_models
    src/models/linear_regression.cpp
)
target_link_libraries(cpp_ml_models PUBLIC
    cpp_ml_core
)

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