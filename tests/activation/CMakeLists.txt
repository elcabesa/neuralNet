target_sources(unit_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/linear-test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/relu-test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/activation-test.cpp
)

# Make sure the compiler can find include files for our libNeuralNet library
# when other libraries or executables link to libNeuralNet
target_include_directories (unit_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

