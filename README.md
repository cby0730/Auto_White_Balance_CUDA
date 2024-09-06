# Auto_White_Balance_CUDA

## Overview
This project implements two white balance methods: von Kries and Gray World. It includes both CPU and GPU (CUDA) implementations of the von Kries method, demonstrating significant performance improvements with GPU acceleration. The project aims to provide efficient and effective white balancing solutions for high-resolution images.

## White Balance Methods

### 1. Von Kries Method
The von Kries method is a chromatic adaptation technique used in color image processing. It works by independently scaling the RGB channels of an image to achieve color constancy.

#### Implementation Details:
- **CPU Version**: Implemented in C++ using OpenCV.
- **GPU Version**: Implemented using CUDA for parallel processing.
- **Performance**: The GPU version is approximately 6 times faster than the CPU version for 32K resolution images.

### 2. Gray World Method
The Gray World algorithm is based on the assumption that the average reflectance in a scene is achromatic. It adjusts the color balance by scaling the RGB channels to have equal average values.

#### Implementation Details:
- Implemented in C++ using OpenCV.
- Currently only available in CPU version.

## Project Structure

```
.
├── cpu_ver
│   ├── main.cpp
│   └── Makefile
├── gpu_ver
│   ├── main.cpp
│   ├── con_Kries_cuda.cu
│   ├── con_Kries_cuda.cuh
│   └── Makefile
├── image
│   ├── img_1.jpg
│   └── ...
└── README.md
```

## Requirements

- OpenCV 4.10
- CUDA Toolkit (for GPU version)
- C++17 compatible compiler (g++-12 recommended)
- CMake (for building)

## Building the Project

**Important**: Ensure that the correct path to the image file is set in the `main.cpp` file before building.

### CPU Version
```bash
cd cpu_ver
make
```

### GPU Version
```bash
cd gpu_ver
make
```

## Usage

After building, you can run the program with:

```bash
./main
```

The program will process the image specified in the `main()` function (default: `"../image/img_7.png"`). It will output timing information and save the processed image as `"Von_Kries.jpg"`.

**Note**: You can also generate a `"gray_world.jpg"` output by uncommenting the Gray World section in the `main()` function.

## Performance Comparison

For a 32K resolution image:
- CPU von Kries: 242976 ms (exact time may vary based on hardware)
- GPU von Kries: 37995 ms (approximately 6.4 times faster than CPU)

These results demonstrate the significant performance improvement achieved by the GPU implementation, especially for high-resolution images.

## Implementation Details

### Von Kries Method (CPU and GPU)

1. Convert the input image from RGB to XYZ color space.
2. Calculate the scaling factors based on the top 5% brightest pixels.
3. Apply the scaling factors to each pixel.
4. Convert the result back to RGB color space.

The GPU implementation uses CUDA to parallelize the scaling operation, resulting in significant speedup for large images.

### Gray World Method

1. Calculate the average value for each color channel.
2. Compute scaling factors to equalize these averages.
3. Apply the scaling factors to each pixel.