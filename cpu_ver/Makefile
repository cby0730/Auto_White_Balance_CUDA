# Compiler
CXX = g++-12

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra

# OpenCV flags and libraries
OPENCV_CFLAGS = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`

# Target executable name
TARGET = main

# Source files
SOURCES = main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Rule to link the program
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(OPENCV_LIBS)

# Rule to compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean run