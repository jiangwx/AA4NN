cmake_minimum_required(VERSION 3.12)
project(Jaffe-AA)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

link_libraries(/usr/local/lib/libaffa.a)
link_libraries(/opt/OpenBLAS/lib/libopenblas.a)
include_directories(/opt/OpenBLAS/include)
link_libraries(pthread)

add_executable(Jaffe-AA main.cpp convolution.cpp convolution_mm.cpp common.cpp CNN.h lrn.cpp pooling.cpp sigmoid.cpp innerproduct.cpp activation.cpp Inception_v1.cpp concat.cpp AA.cpp Darknet19.cpp Darknet19.h)

target_link_libraries(Jaffe-AA ${OpenCV_LIBS})