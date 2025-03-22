set(FFTW_ROOT ${PROJECT_SOURCE_DIR}/src/fftw)
set(FFTW_LIBRARY_DIR ${FFTW_ROOT}/lib)

find_library(FFTW_LIB fftw3f PATHS ${FFTW_LIBRARY_DIR})

add_executable(fft_planner_test
    fft_planner_test.cpp
    ../FFTWRealToComplex.cpp
    ../FFTWComplexToReal.cpp
)
add_executable(test_dfnetwork dfn_test.cpp)

include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/fftw/include
    ${PROJECT_SOURCE_DIR}/src/onnx/include
)

target_link_libraries(test_dfnetwork
    gtest
    gtest_main
    dfnetwork
    dfstate
    dfparams
    runtime_params
    fftwc2r
    fftwr2c
    ${CMAKE_SOURCE_DIR}/src/fftw/lib/libfftw3f.dylib
)

target_link_libraries(fft_planner_test 
    gtest
    gtest_main
    ${CMAKE_SOURCE_DIR}/src/fftw/lib/libfftw3f.dylib
)

target_compile_definitions(test_dfnetwork PRIVATE
    MODEL_PATH="${CMAKE_SOURCE_DIR}/dfn3_onnx/export"
)

add_test(NAME FFTPlannerTest COMMAND fft_planner_test)
add_test(NAME DFNeworkTest COMMAND test_dfnetwork)
include(GoogleTest)
gtest_discover_tests(fft_planner_test)
gtest_discover_tests(test_dfnetwork)
