add_library(libNeuralNet)


add_executable(neuralNet neuralNet.cpp )
target_link_libraries (neuralNet libNeuralNet)

# Make sure the compiler can find include files for our libNeuralNet library
# when other libraries or executables link to libNeuralNet
target_include_directories (libNeuralNet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(activation)
add_subdirectory(cost)
add_subdirectory(layer)
add_subdirectory(input)
add_subdirectory(trainAlgorithms)
add_subdirectory(model)
add_subdirectory(trainSet)

