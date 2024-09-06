#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

cv::Mat autoWhiteBalance(const cv::Mat& input) {
    /**
     * this white balance algorithm is based on gray world assumption
     * it assumes that the average of the R, G, and B channels should be equal
     * so it calculates the average of each channel and scales them accordingly
     * to make the average of all channels equal
     */
    cv::Mat bgr[3];
    cv::split(input, bgr);

    std::vector<double> avgChannels(3);
    
    for (int i = 0; i < 3; i++) {
        std::vector<uchar> channelValues;
        channelValues.assign(bgr[i].begin<uchar>(), bgr[i].end<uchar>());
        std::sort(channelValues.begin(), channelValues.end(), std::greater<uchar>());
        
        int numPixels = static_cast<int>(channelValues.size() * 0.05);
        double sum = std::accumulate(channelValues.begin(), channelValues.begin() + numPixels, 0.0);
        avgChannels[i] = sum / numPixels;
    }

    double maxAvg = *std::max_element(avgChannels.begin(), avgChannels.end());
    std::vector<double> scale = {maxAvg / avgChannels[0], maxAvg / avgChannels[1], maxAvg / avgChannels[2]};

    cv::Mat result;
    input.convertTo(result, CV_32FC3, 1.0 / 255.0);
    
    std::vector<cv::Mat> channels;
    cv::split(result, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = channels[i] * scale[i];
    }
    
    cv::merge(channels, result);
    result.convertTo(result, CV_8UC3, 255.0);
    
    return result;
}

// XYZ到RGB的轉換矩陣
const cv::Matx33f XYZ2RGB(3.2404542, -1.5371385, -0.4985314,
                         -0.9692660,  1.8760108,  0.0415560,
                          0.0556434, -0.2040259,  1.0572252);

// RGB到XYZ的轉換矩陣
const cv::Matx33f RGB2XYZ(0.4124564,  0.3575761,  0.1804375,
                          0.2126729,  0.7151522,  0.0721750,
                          0.0193339,  0.1191920,  0.9503041);

// D65白點
const float D65_y = 250.0f;
const float D65_x = 0.31271f;
const float D65_z = 0.32902f;
const cv::Vec3f D65(D65_y * D65_x / D65_z, D65_y, (1.0f - D65_x - D65_z) * D65_y / D65_z);

cv::Mat vonKries(const cv::Mat& input)
{
    //CV_Assert(input.type() == CV_8UC3);

    // 將圖像轉換為浮點型
    cv::Mat floatImage;
    input.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    // 將RGB轉換為XYZ
    cv::Mat XYZ;
    cv::transform(floatImage, XYZ, RGB2XYZ);

    cv::Mat channals[3];
    cv::split(XYZ, channals);

    // 找到XYZ加起來最大的錢5%的像素
    std::vector<float> avgChannels(3);
    int totalPixels = input.rows * input.cols;
    int numPixels = static_cast<int>(totalPixels * 0.05);
    for (int i = 0; i < 3 ; ++i )
    {
        std::vector<uchar> channalValues;
        channalValues.assign(channals[i].begin<uchar>(), channals[i].end<uchar>());
        std::nth_element(channalValues.begin(), channalValues.begin() + numPixels, channalValues.end(), std::greater<uchar>());
        //std::sort(channalValues.begin(), channalValues.end(), std::greater<uchar>());

        // int numPixels = static_cast<int>(channalValues.size() * 0.05);
        float sum = std::accumulate(channalValues.begin(), channalValues.begin() + numPixels, 0.0);
        avgChannels[i] = sum / numPixels;
    }

    cv::Vec3f K_xyz(
        D65[0] / avgChannels[0],
        D65[1] / avgChannels[1],
        D65[2] / avgChannels[2]
    );

    auto start = std::chrono::high_resolution_clock::now();
    // 進行von Kries轉換
    cv::Mat XYZ_vonKries = XYZ.clone();
    for (int i = 0; i < XYZ.rows; ++i)
    {
        for (int j = 0; j < XYZ.cols; ++j)
        {
            cv::Vec3f& pixel = XYZ_vonKries.at<cv::Vec3f>(i, j);
            pixel[0] *= K_xyz[0];
            pixel[1] *= K_xyz[1];
            pixel[2] *= K_xyz[2];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU von kries Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // 將XYZ轉換為RGB
    cv::Mat RGB_vonKries;
    cv::transform(XYZ_vonKries, RGB_vonKries, XYZ2RGB);

    // 將浮點型圖像轉換為8位元
    cv::Mat result;
    RGB_vonKries.convertTo(result, CV_8UC3, 255.0);

    return result;
}

int main() {
    cv::Mat image = cv::imread("../image/img_7.png");
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    //cv::Mat whiteBalanced = autoWhiteBalance(image);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    //cv::imwrite("Gray_world.jpg", whiteBalanced);
    

    start = std::chrono::high_resolution_clock::now();
    cv::Mat vonKriesImage = vonKries(image);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    cv::imwrite("Von_Kries.jpg", vonKriesImage);

    return 0;
}