#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <string>

using namespace std;

double computeCovDet(const cv::Mat& image, int kernelSize, int x, int y) {
    int half = kernelSize / 2;

    int startX = max(0, x - half);
    int startY = max(0, y - half);
    int endX = min(image.cols, x + half);
    int endY = min(image.rows, y + half);

    // Extract the region
    cv::Mat region = image(cv::Rect(startX, startY, endX - startX, endY - startY)).clone();
    cv::Mat reshapedNeighborhood = region.reshape(1, region.total());

    int n = region.rows * region.cols;
    cv::Mat samples(n, 3, CV_64F);

    cv::Mat mean, covar;
    cv::calcCovarMatrix(reshapedNeighborhood, covar, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS, CV_64F);

    return cv::determinant(covar);
}

cv::Mat denoisingGaussianFilter(const cv::Mat& src, int neighborhoodSize, double factorRatio) {
    int half = neighborhoodSize / 2;
    cv::Mat result = src.clone();

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            double covDet = computeCovDet(src, neighborhoodSize, x, y);

            // cout << "CovDet for pixel (" << x << ", " << y << "): " << covDet << endl;

            int ksize = neighborhoodSize;
            if (covDet != 0){
                int ksize = static_cast<int>(std::max(3.0, factorRatio / (covDet)));
                if (ksize % 2 == 0) ksize += 1;
            }

            // cout << "Adaptive kernel size for pixel (" << x << ", " << y << "): " << ksize << endl;

            int startX = std::max(0, x - half);
            int startY = std::max(0, y - half);
            int endX = std::min(src.cols, x + half);
            int endY = std::min(src.rows, y + half);
            cv::Mat region = src(cv::Rect(startX, startY, endX - startX, endY - startY));

            cv::Mat blurred;
            cv::GaussianBlur(region, blurred, cv::Size(ksize, ksize), 0);

            int centerY = (y - startY);
            int centerX = (x - startX);
            result.at<cv::Vec3b>(y, x) = blurred.at<cv::Vec3b>(centerY, centerX);
        }
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: ./denoising image.jpg neighborhoodSize factorRatio\n";
        return -1;
    }

    string imgPath = argv[1];
    int neighborhoodSize = atoi(argv[2]);
    double factorRatio = atof(argv[3]);

    cv::Mat image = cv::imread(imgPath);
    if (image.empty()) {
        cerr << "Failed to open image.\n";
        return -1;
    }

    if (neighborhoodSize % 2 == 0) {
        cout << "Kernel size must be odd " << endl;
        return -1;
    }
    if (factorRatio <= 0) {
        cout << "Sigma must be greater than 0" << endl;
        return -1;
    }

    auto begin = chrono::high_resolution_clock::now();

    cv::Mat result = denoisingGaussianFilter(image, neighborhoodSize, factorRatio);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - begin;
    cout << "Total time: " << diff.count() << " s" << endl;

    cv::imshow("Original", image);
    cv::imshow("Denoising Filtered", result);
    cv::imwrite("result_images/denoising.jpg", result);
    cv::waitKey(0);

    return 0;
}
