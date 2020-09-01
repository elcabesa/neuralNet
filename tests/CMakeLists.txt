enable_testing()
# Now simply link against gtest or gtest_main as needed. Eg
add_executable(unit_tests 
    dense-test.cpp
    linear-test.cpp
    relu-test.cpp
    main.cpp
)

target_link_libraries(unit_tests gtest libNeuralNet)
target_include_directories (unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
