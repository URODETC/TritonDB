project(tritondb_lib)

file(GLOB_RECURSE SRCS src/*.cpp)
add_library(tritondb_lib STATIC ${SRCS})
target_include_directories(tritondb_lib PUBLIC src)


file(GLOB_RECURSE TEST_SRCS test/*.cpp)

add_executable(tritondb_tests ${TEST_SRCS})
target_link_libraries(tritondb_tests gtest gtest_main tritondb_lib)