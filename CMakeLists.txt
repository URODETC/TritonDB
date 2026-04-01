cmake_minimum_required(VERSION 4.3)
project(tritondb)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(lib)

file(GLOB_RECURSE SRCS src/*.cpp)
add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_link_libraries(${PROJECT_NAME} PUBLIC tritondb_lib)
