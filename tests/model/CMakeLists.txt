target_sources(unit_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/modelTest.cpp
)

# Make sure the compiler can find include files for our libNeuralNet library
# when other libraries or executables link to libNeuralNet
target_include_directories (unit_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

