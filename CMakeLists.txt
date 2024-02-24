cmake_minimum_required(VERSION 3.22.1)
project(neural_network)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main src/main.cc)

target_include_directories(main PUBLIC include)

