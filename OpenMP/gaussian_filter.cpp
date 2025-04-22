#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <string>

using namespace std;

// Function to generate a Gaussian kernel
cv::Mat createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    cv::Mat kernel(ksize, ksize, CV_64F);
    double sum = 0.0;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma));
            value /= 2 * CV_PI * sigma * sigma;
            kernel.at<double>(y + half, x + half) = value;
            sum += value;
        }
    }

    // Normalize the kernel
    kernel /= sum;
    return kernel;
}

// Apply manual convolution with Gaussian kernel
cv::Mat applyGaussianFilter(const cv::Mat& src, int ksize, double sigma) {
    cv::Mat kernel = createGaussianKernel(ksize, sigma);

    int half = ksize / 2;
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, half, half, half, half, cv::BORDER_REFLECT);

    cv::Mat output = cv::Mat::zeros(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < src.channels(); c++) {
                double sum = 0.0;
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int iy = y + ky + half;
                        int ix = x + kx + half;
                        double pixel = static_cast<double>(padded.at<cv::Vec3b>(iy, ix)[c]);
                        double weight = kernel.at<double>(ky + half, kx + half);
                        sum += pixel * weight;
                    }
                }
                output.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(round(sum));
            }
        }
    }
    return output;
}

cv::Mat generateAnaglyph(const cv::Mat& source, const string& anaglyph_type, int iter) {
    // Validate input dimensions
    if (source.cols % 2 != 0) {
        throw runtime_error("Source image width must be even to split into stereo pair.");
    }

    cv::Mat leftImage = source(cv::Rect(0, 0, source.cols / 2, source.rows));
    cv::Mat rightImage = source(cv::Rect(source.cols / 2, 0, source.cols / 2, source.rows));

    cv::Mat anaglyph(leftImage.rows, leftImage.cols, CV_8UC3);

    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                const cv::Vec3b& leftPixel = leftImage.at<cv::Vec3b>(i, j);
                const cv::Vec3b& rightPixel = rightImage.at<cv::Vec3b>(i, j);

                if (anaglyph_type == "true") {
                    anaglyph.at<cv::Vec3b>(i, j)[0] = 0.299 * rightPixel[0] + 0.578 * rightPixel[1] + 0.114 * rightPixel[2];
                    anaglyph.at<cv::Vec3b>(i, j)[1] = 0;
                    anaglyph.at<cv::Vec3b>(i, j)[2] = 0.299 * leftPixel[0] + 0.578 * leftPixel[1] + 0.114 * leftPixel[2];
                }
                else if (anaglyph_type == "gray") {
                    anaglyph.at<cv::Vec3b>(i, j)[0] = 0.299 * rightPixel[0] + 0.578 * rightPixel[1] + 0.114 * rightPixel[2];
                    anaglyph.at<cv::Vec3b>(i, j)[1] = 0.299 * rightPixel[0] + 0.578 * rightPixel[1] + 0.114 * rightPixel[2];
                    anaglyph.at<cv::Vec3b>(i, j)[2] = 0.299 * leftPixel[0] + 0.578 * leftPixel[1] + 0.114 * leftPixel[2];
                }
                else if (anaglyph_type == "color") {
                    anaglyph.at<cv::Vec3b>(i, j)[0] = leftPixel[0];
                    anaglyph.at<cv::Vec3b>(i, j)[1] = rightPixel[1];
                    anaglyph.at<cv::Vec3b>(i, j)[2] = rightPixel[2];
                }
                else if (anaglyph_type == "half_color") {
                    anaglyph.at<cv::Vec3b>(i, j)[0] = 0.299 * leftPixel[0] + 0.578 * leftPixel[1] + 0.114 * leftPixel[2];
                    anaglyph.at<cv::Vec3b>(i, j)[1] = rightPixel[1];
                    anaglyph.at<cv::Vec3b>(i, j)[2] = rightPixel[2];
                }
                else if (anaglyph_type == "optimized") {
                    anaglyph.at<cv::Vec3b>(i, j)[0] = 0.7 * leftPixel[1] + 0.3 * leftPixel[2];
                    anaglyph.at<cv::Vec3b>(i, j)[1] = rightPixel[1];
                    anaglyph.at<cv::Vec3b>(i, j)[2] = rightPixel[2];
                }                
            }
        }
    }

    return anaglyph;
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        cout << "Usage: ./anaglyph image.jpg anaglyph_type kernel_size sigma" << endl;
        return -1;
    }

    cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (source.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    string anaglyph_type = argv[2];
    vector<string> anaglyph_types = {"true", "gray", "color", "half_color", "optimized"};
    if (find(anaglyph_types.begin(), anaglyph_types.end(), anaglyph_type) == anaglyph_types.end()) {
        cout << "Invalid anaglyph type! Use 'true', 'gray', 'color', 'half_color', or 'optimized'." << endl;
        return -1;
    }

    auto begin = chrono::high_resolution_clock::now();

    cv::Mat anaglyph;
    const int iter = 10000;
    try {
        anaglyph = generateAnaglyph(source, anaglyph_type, iter);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - begin;

    // Apply manual Gaussian filter
    int ksize = atoi(argv[3]);
    double sigma = atof(argv[4]);
    if (ksize % 2 == 0) {
        cout << "Kernel size must be odd " << endl;
        return -1;
    }
    if (sigma <= 0 || sigma > 10) {
        cout << "Sigma must be in range 0.1 and 10" << endl;
        return -1;
    }
    
    cv::Mat result = applyGaussianFilter(anaglyph, ksize, sigma);
    
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // cv::imshow("Original", source);
    cv::imshow("Manual Gaussian Filtered", result);
    cv::imwrite("result_images/gaussian_filter.jpg", result);

    cv::waitKey();
    return 0;
}
