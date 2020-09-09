enable_testing()
# Now simply link against gtest or gtest_main as needed. Eg
add_executable(unit_tests 
    main.cpp
)

target_link_libraries(unit_tests gtest libNeuralNet)
target_include_directories (unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(activation)
add_subdirectory(cost)
add_subdirectory(input)
add_subdirectory(layer)
add_subdirectory(model)
