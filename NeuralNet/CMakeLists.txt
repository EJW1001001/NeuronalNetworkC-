cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Incluir directorios
include_directories(include)
include_directories(include/neuralnet)

# Archivos fuente
set(SOURCES
    src/Neuron.cpp
    src/Layer.cpp
    src/Activations.cpp
    src/LossFunctions.cpp
    src/Optimizers.cpp
    src/Model.cpp
)

# Archivos de cabecera
set(HEADERS
    include/neuralnet/Neuron.h
    include/neuralnet/Layer.h
    include/neuralnet/Activations.h
    include/neuralnet/LossFunctions.h
    include/neuralnet/Optimizers.h
    include/neuralnet/Model.h
)

# Añadir ejecutable para ejemplos
add_executable(xor_example examples/xor_example.cpp ${SOURCES} ${HEADERS})

# Añadir ejecutable para pruebas
add_executable(test_neuralnet tests/test_neuralnet.cpp ${SOURCES} ${HEADERS})
