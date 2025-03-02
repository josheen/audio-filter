cmake_minimum_required(VERSION 3.16)

project(AudioFilter VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

ADD_CUSTOM_TARGET(link_target ALL COMMAND ${CMAKE_COMMAND} -E create_symlink 
    compile_commands.json ${PROJECT_SOURCE_DIR}/compile_commands.json)


add_library(Utils src/utils.cpp)
target_include_directories(Utils PUBLIC ${PROJECT_SOURCE_DIRECTORY}/include)

add_executable(AudioFilter src/main.cpp)
target_link_libraries(AudioFilter Utils)

enable_testing()
add_executable(TestsUtils tests/test_utils.cpp)
target_link_libraries(TestsUtils Utils)
add_test(NAME TestUtils COMMAND TestUtils)
