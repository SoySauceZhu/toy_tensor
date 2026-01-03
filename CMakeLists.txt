cmake_minimum_required(VERSION 3.16)
project(mytensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mytensor_static STATIC
    src/tensor.cpp
        src/autograd/autograd.cpp
    src/engine.cpp
    src/ops.cpp
)

target_include_directories(mytensor_static
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(mytensor_demo src/main.cpp
        src/autograd/addnode.cpp)
target_link_libraries(mytensor_demo PRIVATE mytensor_static)
