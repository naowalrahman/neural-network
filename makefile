# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -std=c++11

# Executable name
EXEC = Main

# Source and header files
SRC_FILES = Activations.cpp Layer.cpp Linalg.cpp Main.cpp NeuralNetwork.cpp Neuron.cpp
OBJ_FILES = $(SRC_FILES:.cpp=.o)
HEADER_FILES = Activations.hpp Layer.hpp Linalg.hpp NeuralNetwork.hpp Neuron.hpp

# Default target
all: $(EXEC)

# Rule to create the executable
$(EXEC): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJ_FILES)

# Rule to compile .cpp files to .o files
%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXEC)

# Phony targets
.PHONY: all clean
