# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NeuralNetworkProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-Wall -O2 -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -g -ggdb -D_FORTIFY_SOURCE=2 -fsanitize=address -fsanitize=undefined -fno-sanitize-recover -fstack-protector")
else()
    set(CMAKE_CXX_FLAGS "-O2")
endif()

# Add the executable
add_executable(NeuralNetwork
    Main.cpp
    Activations.cpp
    Layer.cpp
    Linalg.cpp
    NeuralNetwork.cpp
    Neuron.cpp
)

# Add include directories
target_include_directories(NeuralNetwork PUBLIC
    ${PROJECT_SOURCE_DIR}
)
