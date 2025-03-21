add_executable(fft_planner_test
    fft_planner_test.cpp
    ../FFTWRealToComplex.cpp
    ../FFTWComplexToReal.cpp
)

target_include_directories(fft_planner_test PRIVATE 
    ../include
    ../fftw/include
)
target_link_libraries(fft_planner_test 
    gtest
    gtest_main
    ${CMAKE_SOURCE_DIR}/src/fftw/lib/libfftw3f.dylib
)

add_test(NAME FFTPlannerTest COMMAND fft_planner_test)
include(GoogleTest)
gtest_discover_tests(fft_planner_test)
