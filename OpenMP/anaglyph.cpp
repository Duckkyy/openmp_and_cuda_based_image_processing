#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "Usage: ./anaglyph image.jpg anaglyph_type" << endl;
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

    // Simulate stereo pair by shifting left and right
    cv::Mat leftImage = source(cv::Rect(0, 0, source.cols / 2, source.rows));
    cv::Mat rightImage = source(cv::Rect(source.cols / 2, 0, source.cols / 2, source.rows));

    cv::Mat anaglyph(leftImage.rows, leftImage.cols, CV_8UC3);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 500;

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

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - begin;

    cv::imshow("Anaglyph Image", anaglyph);
    if (anaglyph_type == "true") {
        cv::imwrite("result_images/true_anaglyph.jpg", anaglyph);
    }
    else if (anaglyph_type == "gray") {
        cv::imwrite("result_images/gray_anaglyph.jpg", anaglyph);
    }
    else if (anaglyph_type == "color") {
        cv::imwrite("result_images/color_anaglyph.jpg", anaglyph);
    }
    else if (anaglyph_type == "half_color") {
        cv::imwrite("result_images/half_color_anaglyph.jpg", anaglyph);
    }
    else if (anaglyph_type == "optimized") {
        cv::imwrite("result_images/optimized_anaglyph.jpg", anaglyph);
    }
    
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::waitKey();
    return 0;
}
