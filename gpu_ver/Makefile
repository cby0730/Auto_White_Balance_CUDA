CXX = g++-12
NVCC = nvcc

CXXFLAGS = -g `pkg-config --cflags opencv4` -I/usr/local/cuda/include
LDFLAGS = `pkg-config --libs opencv4` -lcudart -L/usr/local/cuda/lib64
CUDA_FLAGS = -I/usr/local/cuda/include `pkg-config --cflags opencv4`

SOURCES = $(wildcard *.cpp)
CUDA_SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

TARGET = main

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET)

.PHONY: clean