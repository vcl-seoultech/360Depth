#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>
#include <opencv2/opencv.hpp>

#include "Coordinate.hpp"
#include "type.hpp"

using namespace std;

namespace DEMO
{

// rotate image
void rotateImg(const cv::Mat &src, cv::Mat &dst, const cv::Mat &Rmat);
void rotateImg(const cv::Mat &src, cv::Mat &dst, const cv::Mat &Rmat, DEMO::Pixel method);
void rotateImg(const cv::Mat &src, cv::Mat &dst, const double ax, const double ay, const double az);
// compose rotation matrix
void composeRotationMat(cv::Mat &R, const double x, const double y, const double z);

double calcDistance(dPair a, dPair b);

} //namespace DEMO

#endif // FUNCTIONS_H