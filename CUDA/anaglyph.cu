#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <string>
#include <cmath>
#include <chrono>

#include "helper_math.h"

using namespace std;

// Enum for different anaglyph types
enum AnaglyphType {
    NORMAL = 0,
    TRUE,
    GRAY,
    COLOR,
    HALFCOLOR,
    OPTIMIZED
};

// CUDA kernel for processing anaglyphs
__global__ void processKernel(const cv::cuda::PtrStep<uchar3> left_image,
                              const cv::cuda::PtrStep<uchar3> right_image,
                              cv::cuda::PtrStep<uchar3> anaglyph_image,
                              int rows,
                              int cols,
                              int anaglyph_type) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        uchar3 left_pixel = left_image(y, x);
        uchar3 right_pixel = right_image(y, x);

        uchar3 result_pixel;

        if (anaglyph_type == TRUE) {
            result_pixel = make_uchar3(
                0.299f * right_pixel.z + 0.578f * right_pixel.y + 0.114f * right_pixel.x,
                0,
                0.299f * left_pixel.z + 0.578f * left_pixel.y + 0.114f * left_pixel.x
            );
        } else if (anaglyph_type == GRAY) {
            float gray_right = 0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z;
            float gray_left = 0.299f * left_pixel.x + 0.578f * left_pixel.y + 0.114f * left_pixel.z;
            result_pixel = make_uchar3(gray_right, gray_right, gray_left);
        } else if (anaglyph_type == COLOR) {
            result_pixel = make_uchar3(right_pixel.x, right_pixel.y, left_pixel.z);
        } else if (anaglyph_type == HALFCOLOR) {
            result_pixel = make_uchar3(
                0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                right_pixel.y,
                left_pixel.z
            );
        } else if (anaglyph_type == OPTIMIZED) {
            result_pixel = make_uchar3(
                0.7f * right_pixel.y + 0.3f * right_pixel.x,
                right_pixel.y,
                left_pixel.z
            );
        } else {
            result_pixel = left_pixel;
        }

        anaglyph_image(y, x) = result_pixel;
    }
}

// Utility function to calculate grid size
int divUp(int a, int b) {
    return (a + b - 1) / b;
}

// Function to process anaglyphs using CUDA
void processCUDA(const cv::cuda::GpuMat& d_left_image,
                 const cv::cuda::GpuMat& d_right_image,
                 cv::cuda::GpuMat& d_anaglyph_image,
                 int rows,
                 int cols,
                 int anaglyph_type) {
    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    processKernel<<<grid, block>>>(d_left_image, d_right_image, d_anaglyph_image, rows, cols, anaglyph_type);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_path> <anaglyph_type>" << endl;
        return -1;
    }

    // Load stereo image
    cv::Mat stereo_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (stereo_image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    // Parse anaglyph type
    int anaglyph_type = atoi(argv[2]);
    if (anaglyph_type < NORMAL || anaglyph_type > OPTIMIZED) {
        cerr << "Error: Invalid anaglyph type." << endl;
        return -1;
    }

    // Split stereo image into left and right images
    cv::Mat left_image(stereo_image, cv::Rect(0, 0, stereo_image.cols / 2, stereo_image.rows));
    cv::Mat right_image(stereo_image, cv::Rect(stereo_image.cols / 2, 0, stereo_image.cols / 2, stereo_image.rows));
    cv::Mat anaglyph_image;

    // Upload images to GPU
    cv::cuda::GpuMat d_left_image, d_right_image, d_anaglyph_image;
    d_left_image.upload(left_image);
    d_right_image.upload(right_image);
    d_anaglyph_image.create(left_image.size(), left_image.type());

    // Measure performance
    const int iterations = 100;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        processCUDA(d_left_image, d_right_image, d_anaglyph_image, left_image.rows, left_image.cols, anaglyph_type);
        d_anaglyph_image.download(anaglyph_image);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    // Save the result
    string output_filename = "results/anaglyph_result.jpg";
    cv::imwrite(output_filename, anaglyph_image);

    // Display performance metrics
    cout << "Total time for " << iter << " iterations: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    return 0;
}