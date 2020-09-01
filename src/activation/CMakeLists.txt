target_sources(libNeuralNet PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/relu.cpp
    )

# Make sure the compiler can find include files for our libNeuralNet library
# when other libraries or executables link to libNeuralNet
target_include_directories (libNeuralNet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

