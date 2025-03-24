cmake_minimum_required(VERSION 3.16)

project(AudioFilter VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/fftw/include
    ${PROJECT_SOURCE_DIR}/src/onnx/include
    ${PROJECT_SOURCE_DIR}/third_party/eigen
    ${PROJECT_SOURCE_DIR}/third_party/portaudio
    ${PROJECT_SOURCE_DIR}/third_party/libsndfile/include
)

set(FFTW_ROOT ${PROJECT_SOURCE_DIR}/src/fftw)
set(FFTW_LIBRARY_DIR ${FFTW_ROOT}/lib)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR}/src/onnx)
set(ONNX_LIBRARY_DIR ${ONNX_ROOT}/lib)

find_library(FFTW_LIB fftw3f PATHS ${FFTW_LIBRARY_DIR})
find_path(FFTW_INCLUDE_DIR fftw3.h PATHS ${FFTW_INCLUDE_DIR})

find_library(ONNX_LIB onnxruntime PATHS ${ONNX_LIBRARY_DIR})
add_subdirectory(third_party/portaudio external/libsndfile)

add_executable(AudioFilter src/main.cpp)

add_library(fftwc2r src/FFTWComplexToReal.cpp)
add_library(fftwr2c src/FFTWRealtoComplex.cpp)
add_library(dfstate src/DFState.cpp)
add_library(dfparams src/DFParams.cpp)
add_library(dfnetwork src/DFNetwork.cpp)
target_link_libraries(dfnetwork ${ONNX_LIB})
add_library(runtime_params src/RuntimeParams.cpp)

target_link_libraries(dfstate ${FFTW_LIB})

target_link_libraries(AudioFilter
    fftwc2r
    fftwr2c
    dfnetwork
    dfstate
    dfparams
    runtime_params
    ${FFTW_LIB}
    ${ONNX_LIB}
    portaudio
    ${PROJECT_SOURCE_DIR}/src/lib/libsndfile.dylib
)
target_compile_definitions(AudioFilter PRIVATE
    MODEL_PATH="${CMAKE_SOURCE_DIR}/dfn3_onnx/export"
)

ADD_CUSTOM_TARGET(link_target ALL COMMAND ${CMAKE_COMMAND} -E create_symlink
    compile_commands.json ${PROJECT_SOURCE_DIR}/compile_commands.json)

add_subdirectory(third_party/googletest)
enable_testing()
add_subdirectory(src/tests)
