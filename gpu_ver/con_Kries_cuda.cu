#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

__global__ void vonKriesKernel(float* data, int width, int height, float3 K_xyz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height)
    {
        int i = (idy * width + idx) * 3;
        data[i] *= K_xyz.x;
        data[i + 1] *= K_xyz.y;
        data[i + 2] *= K_xyz.z;
    }
}

void launchVonKriesKernel(cv::Mat& XYZ_vonKries, const cv::Vec3f& K_xyz)
{
    float* d_data;
    int width = XYZ_vonKries.cols;
    int height = XYZ_vonKries.rows;
    size_t size = width * height * 3 * sizeof(float);

    // 分配裝置記憶體
    cudaMalloc(&d_data, size);

    // 將資料複製到裝置
    cudaMemcpy(d_data, XYZ_vonKries.data, size, cudaMemcpyHostToDevice);

    // 設置網格和區塊尺寸
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 啟動核心
    vonKriesKernel<<<numBlocks, threadsPerBlock>>>(d_data, width, height, make_float3(K_xyz[0], K_xyz[1], K_xyz[2]));

    // 將結果複製回主機
    cudaMemcpy(XYZ_vonKries.data, d_data, size, cudaMemcpyDeviceToHost);

    // 釋放裝置記憶體
    cudaFree(d_data);
}