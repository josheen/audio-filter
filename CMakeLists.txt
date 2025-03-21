cmake_minimum_required(VERSION 3.16)

project(AudioFilter VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/src/include)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(FFTW_ROOT ${PROJECT_SOURCE_DIR}/src/fftw)
set(FFTW_INCLUDE_DIR ${FFTW_ROOT}/include)
set(FFTW_LIBRARY_DIR ${FFTW_ROOT}/lib)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR}/src/onnx)
set(ONNX_INCLUDE_DIR ${ONNX_ROOT}/include)
set(ONNX_LIBRARY_DIR ${ONNX_ROOT}/lib)

find_library(FFTW_LIB fftw3f PATHS ${FFTW_LIBRARY_DIR})
find_path(FFTW_INCLUDE_DIR fftw3.h PATHS ${FFTW_INCLUDE_DIR})

find_library(ONNX_LIB onnxruntime PATHS ${ONNX_LIBRARY_DIR})
find_path(ONNX_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNX_INCLUDE_DIR})
ADD_CUSTOM_TARGET(link_target ALL COMMAND ${CMAKE_COMMAND} -E create_symlink 
    compile_commands.json ${PROJECT_SOURCE_DIR}/compile_commands.json)

add_executable(AudioFilter src/main.cpp)

add_library(fftwc2r src/FFTWComplexToReal.cpp)
add_library(fftwr2c src/FFTWRealtoComplex.cpp)
add_library(Utils src/df_utils.cpp)
target_include_directories(Utils
    PUBLIC ${PROJECT_SOURCE_DIR}/src/include
    PUBLIC ${ONNX_INCLUDE_DIR}
)
target_include_directories(fftwc2r
    PUBLIC ${PROJECT_SOURCE_DIR}/src/include
    PUBLIC ${FFTW_INCLUDE_DIR}
)

target_include_directories(fftwr2c
    PUBLIC ${PROJECT_SOURCE_DIR}/src/include
    PUBLIC ${FFTW_INCLUDE_DIR}
)

target_link_libraries(Utils
    PRIVATE ${FFTW_LIB}
    PRIVATE ${ONNX_LIB}
)
target_link_libraries(AudioFilter
    Utils
    fftwc2r
    fftwr2c
)

enable_testing()
add_executable(TestsUtils tests/test_utils.cpp)
target_link_libraries(TestsUtils Utils)
add_test(NAME TestUtils COMMAND TestUtils)
