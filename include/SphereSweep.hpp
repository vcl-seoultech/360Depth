#ifndef SPHERESWEEP_H
#define SPHERESWEEP_H

#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "functions.hpp"
#include "type.hpp"

namespace DEMO
{

class SphereSweep
{
public:
    // System
    SphereSweep();
    SphereSweep(const std::vector<cv::Mat> &im, const std::vector<cv::Mat> &t);
    SphereSweep(double min_Depth, double max_Depth, double _interval);
    ~SphereSweep();

    // compute depth
    void run(cv::Mat &dst);

    // set
    void setT(const std::vector<cv::Mat> &t);
    void setFrame(const std::vector<cv::Mat> &im);
    void nextFrame(const std::vector<cv::Mat> &im);

    // set parameters
    void setBlockSize(const int blockSize);
    void setMinDepth(const double minDepth);
    void setMaxDepth(const double maxDepth);
    void setNumDepth(const int numDepth);
    void setInterval(const double interval);
    void setMethod(const DEMO::Pixel method);

private:
    // preprocess
    void padImages();

    // computation
    void computeCost(DEMO::Pixel method);
    void computeWinner();
    double computeWinner(const int r, const int c);
    int computeCost(int r, int c, double virtualDepth);
    dPair getPixel(int r, int c, double virtualDepth, int idx);
    void getColor(cv::Mat &dst, dPair pt, int idx, DEMO::Pixel method);

private:
    std::vector<cv::Mat> im; 
    std::vector<cv::Mat> t; 
    
    int imHeight, imWidth;
    int mHeight, mWidth;
    int padHeight, padWidth;
    int numImages;
    
    // memory allocation
    double ***cost;
    double *winDepth;
    cv::Mat imDepth;

    // parameters
    int blockSize, halfSize;
    double minDepth, maxDepth;
    int numDepth;
    double interval;
    DEMO::Pixel method;
    
};

} //namespace DEMO

#endif // SYSTEM_H
