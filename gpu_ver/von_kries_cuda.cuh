#ifndef VON_KRIES_CUDA_CUH
#define VON_KRIES_CUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

__global__ void vonKriesKernel(float* data, int width, int height, float3 K_xyz);
void launchVonKriesKernel(cv::Mat& XYZ_vonKries, const cv::Vec3f& K_xyz);

#endif // VON_KRIES_CUDA_CUH