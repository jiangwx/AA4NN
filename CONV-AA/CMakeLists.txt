cmake_minimum_required(VERSION 3.12)
project(GEMM)

set(CMAKE_CXX_STANDARD 11)

link_libraries(/opt/OpenBLAS/lib/libopenblas.a)
include_directories(/opt/OpenBLAS/include)

link_libraries(pthread)
link_libraries(/usr/local/lib/libaffa.a)

add_executable(GEMM main.cpp convolution.cpp common.cpp CNN.h AA.cpp)