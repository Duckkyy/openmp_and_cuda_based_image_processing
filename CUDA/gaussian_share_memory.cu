#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <cmath>
#include <chrono>  

using namespace std;

enum AnaglyphType {
    NORMAL=0,
    TRUE,
    GRAY,
    COLOR,
    HALFCOLOR,
    OPTIMIZED
};

// Shared memory size for the block
#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 21
#define SHARED_MEM_SIZE (BLOCK_SIZE + MAX_KERNEL_SIZE - 1)

__global__ void generateGaussianKernelKernel(double* gaussKernel, int kernelSize, double sigma) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < kernelSize && dst_y < kernelSize) {
        int halfKernelSize = kernelSize / 2;
        const double PI = 3.14159265358979323846;

        double lp = 1.0 / (2.0 * PI * sigma * sigma);
        double rp = 1.0 / (2.0 * sigma * sigma);

        double gaussianVal = lp * exp(-((dst_x - halfKernelSize) * (dst_x - halfKernelSize) + (dst_y - halfKernelSize) * (dst_y - halfKernelSize)) * rp);
        gaussKernel[dst_y * kernelSize + dst_x] = gaussianVal;
    }
}

__global__ void applyGaussianBlurKernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> dst, int kernelSize, double* gaussKernel) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uchar3 sharedMem[SHARED_MEM_SIZE][SHARED_MEM_SIZE];

    int halfKernelSize = kernelSize / 2;

    int sharedX = threadIdx.x + halfKernelSize;
    int sharedY = threadIdx.y + halfKernelSize;

    if (dst_x < src.cols && dst_y < src.rows) {
        sharedMem[sharedY][sharedX] = src(dst_y, dst_x);
    } else {
        sharedMem[sharedY][sharedX] = make_uchar3(0, 0, 0);
    }

    if (threadIdx.x < halfKernelSize) {
        sharedMem[sharedY][sharedX - halfKernelSize] = src(dst_y, dst_x - halfKernelSize);
        sharedMem[sharedY][sharedX + blockDim.x] = src(dst_y, dst_x + blockDim.x);
    }

    if (threadIdx.y < halfKernelSize) {
        sharedMem[sharedY - halfKernelSize][sharedX] = src(dst_y - halfKernelSize, dst_x);
        sharedMem[sharedY + blockDim.y][sharedX] = src(dst_y + blockDim.y, dst_x);
    }

    if (threadIdx.y < halfKernelSize && threadIdx.x < halfKernelSize) {
        sharedMem[sharedY - halfKernelSize][sharedX - halfKernelSize] = src(dst_y - halfKernelSize, dst_x - halfKernelSize);
        sharedMem[sharedY - halfKernelSize][sharedX + blockDim.x] = src(dst_y - halfKernelSize, dst_x + blockDim.x);
        sharedMem[sharedY + blockDim.y][sharedX - halfKernelSize] = src(dst_y + blockDim.y, dst_x - halfKernelSize);
        sharedMem[sharedY + blockDim.y][sharedX + blockDim.x] = src(dst_y + blockDim.y, dst_x + blockDim.x);
    }
    __syncthreads();

    if (dst_x < src.cols && dst_y < src.rows) {
        double sum[3] = {0.0, 0.0, 0.0};
        double gaussianTotal = 0.0;

        for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
            for (int j = -halfKernelSize; j <= halfKernelSize; ++j) {
                uchar3 pixel = sharedMem[sharedY + i][sharedX + j];
                double gaussianVal = gaussKernel[(i + halfKernelSize) * kernelSize + (j + halfKernelSize)];
                gaussianTotal += gaussianVal;

                double pixelVec[3] = {static_cast<double>(pixel.x), static_cast<double>(pixel.y), static_cast<double>(pixel.z)};
                for (int k = 0; k < 3; ++k) {
                    sum[k] += pixelVec[k] * gaussianVal;
                }
            }
        }

        for (int k = 0; k < 3; ++k) {
            sum[k] /= gaussianTotal;
        }
        dst(dst_y, dst_x) = make_uchar3(static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]));
    }
}

__global__ void processKernel(const cv::cuda::PtrStepSz<uchar3> left_image,
                                     const cv::cuda::PtrStepSz<uchar3> right_image,
                                     cv::cuda::PtrStepSz<uchar3> anaglyph_image,
                                     int anaglyph_type) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < left_image.cols && y < left_image.rows) {
        uchar3 left_pixel = left_image(y, x);
        uchar3 right_pixel = right_image(y, x);

        if (anaglyph_type == TRUE) {
            anaglyph_image(y, x) = make_uchar3(
                0.299f * right_pixel.z + 0.578f * right_pixel.y + 0.114f * right_pixel.x,
                0,
                0.299f * left_pixel.z + 0.578f * left_pixel.y + 0.114f * left_pixel.x
            );
            return;
        }
        else if (anaglyph_type == GRAY) {
            anaglyph_image(y, x) = make_uchar3(
                0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                0.299f * left_pixel.x + 0.578f * left_pixel.y + 0.114f * left_pixel.z
            );
            return;
        }
        else if (anaglyph_type == COLOR) {
            anaglyph_image(y, x) = make_uchar3(
                right_pixel.x,
                right_pixel.y,
                left_pixel.z
            );
            return;
        }
        else if (anaglyph_type == HALFCOLOR) {
            anaglyph_image(y, x) = make_uchar3(
                0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                right_pixel.y,
                left_pixel.z
            );
            return;
        }
        else if (anaglyph_type == OPTIMIZED) {
            anaglyph_image(y, x) = make_uchar3(
                0.7f * right_pixel.y + 0.3f * right_pixel.x,
                right_pixel.y,
                left_pixel.z
            );
            return;
        }
        else {
            anaglyph_image(y, x) = left_pixel;
            return;
        }        
    }
}

__global__ void mergeImagesKernel(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> resultImage, int rows, int cols) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_y < rows && dst_x < cols) {
        resultImage(dst_y, dst_x) = leftImage(dst_y, dst_x);
        resultImage(dst_y, dst_x + cols) = rightImage(dst_y, dst_x);
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(const cv::cuda::GpuMat& d_left_image,
                const cv::cuda::GpuMat& d_right_image,
                cv::cuda::GpuMat& d_anaglyph_image,
                cv::cuda::GpuMat& d_blurred_image,
                int kernelSize,
                int anaglyph_type,
                double* gaussKernel) {
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(d_right_image.cols, block.x), divUp(d_right_image.rows, block.y));
    int rows = d_anaglyph_image.rows;
    int cols = d_anaglyph_image.cols;

    // Apply Gaussian blur kernel with shared memory
    applyGaussianBlurKernel<<<grid, block>>>(d_left_image, d_left_image, kernelSize, gaussKernel);
    applyGaussianBlurKernel<<<grid, block>>>(d_right_image, d_right_image, kernelSize, gaussKernel);

    mergeImagesKernel<<<grid, block>>>(d_left_image, d_right_image, d_blurred_image, rows, cols);

    if (anaglyph_type == NORMAL) {
        d_anaglyph_image = d_left_image;
        return;
    }
    // Create anaglyph image kernel
    processKernel<<<grid, block>>>(d_left_image, d_right_image, d_anaglyph_image, anaglyph_type);
}

int main( int argc, char** argv )
{
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <image_path> <anaglyph_type> <kernel_size> <sigma>" << endl;
        return -1;
    }

    cv::Mat stereo_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (stereo_image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    AnaglyphType anaglyph_type = static_cast<AnaglyphType>(atoi(argv[2]));
    if (anaglyph_type < NORMAL || anaglyph_type > OPTIMIZED) {
        cerr << "Error: Invalid anaglyph type." << endl;
        cerr << "Anaglyph types:" << endl;
        cerr << "0: None Anaglyphs" << endl;
        cerr << "1: True Anaglyphs" << endl;
        cerr << "2: Gray Anaglyphs" << endl;
        cerr << "3: Color Anaglyphs" << endl;
        cerr << "4: Half Color Anaglyphs" << endl;
        cerr << "5: Optimized Anaglyphs" << endl;
        return -1;
    }
    std::string anaglyph_name;

    if (anaglyph_type == TRUE) {
        anaglyph_name = "True";
    } else if (anaglyph_type == GRAY) {
        anaglyph_name = "Gray";
    } else if (anaglyph_type == COLOR) {
        anaglyph_name = "Color";
    } else if (anaglyph_type == HALFCOLOR) {
        anaglyph_name = "Half Color";
    } else if (anaglyph_type == OPTIMIZED) {
        anaglyph_name = "Optimized";
    } else {
        anaglyph_name = "None";
    }

    cv::Mat left_image(stereo_image, cv::Rect(0, 0, stereo_image.cols / 2, stereo_image.rows));
    cv::Mat right_image(stereo_image, cv::Rect(stereo_image.cols / 2, 0, stereo_image.cols / 2, stereo_image.rows));

    int kernelSize = atoi(argv[3]);
    double sigma = atof(argv[4]);
    if (!kernelSize|| !sigma) {
        cerr << "Error: Invalid kernel size or sigma." << endl;
        cerr << "Input kernel size in range odd numbers from 3 to 21" << endl;
        cerr << "Input sigma in range odd numbers from 0.1 to 10" << endl;
        return -1;
    }

    double* gaussKernel;
    cudaMalloc(&gaussKernel, kernelSize * kernelSize * sizeof(double));
    cudaMemset(gaussKernel, 0, kernelSize * kernelSize * sizeof(double));

    dim3 blockSize(16, 16);
    dim3 gridSize((kernelSize + blockSize.x - 1) / blockSize.x, (kernelSize + blockSize.y - 1) / blockSize.y);

    generateGaussianKernelKernel<<<gridSize, blockSize>>>(gaussKernel, kernelSize, sigma);

    cv::Mat anaglyph_image, blurred_image;

    cv::cuda::GpuMat d_left_image, d_right_image, d_blurred_image, d_anaglyph_image;

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 1000;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        d_left_image.upload(left_image);
        d_right_image.upload(right_image);
        d_anaglyph_image.upload(right_image);
        d_blurred_image.upload(stereo_image);
        processCUDA(d_left_image, d_right_image, d_anaglyph_image, d_blurred_image, kernelSize, anaglyph_type, gaussKernel);
        d_blurred_image.download(blurred_image);
        d_anaglyph_image.download(anaglyph_image);
    }

    // Stop the timer
    auto end = chrono::high_resolution_clock::now();

    // Calculate the time difference
    chrono::duration<double> diff = end - begin;

    // Save the anaglyph image
    std::string filename =  "results/share_memory/" + anaglyph_name + "Anaglyph-blurred.jpg";
    cv::imwrite(filename, anaglyph_image);

    std::string blurred_img_name =  "results/share_memory/blurred.jpg";
    cv::imwrite(blurred_img_name, blurred_image);

    // Display performance metrics
    cout << "Total time for " << iter << " iterations: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::waitKey();

    cudaFree(gaussKernel);

    return 0;
}
