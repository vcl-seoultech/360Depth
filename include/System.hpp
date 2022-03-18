#ifndef SYSTEM_H
#define SYSTEM_H

#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
/////////////////////////////////////////////////
#include <cmath>
/////////////////////////////////////////////////

#include "SPHORB/SPHORB.h"
#include "SPHORB/utility.h"

#include "Coordinate.hpp"
#include "SfM.hpp"
#include "SphereSweep.hpp"
#include "functions.hpp"
#include "type.hpp"

namespace DEMO
{

class Coordinate;
class SfM;

class System
{
public:
    // System
    System(const vector<cv::Mat> &im, int idx, double max_Depth, double interval, bool _rigid);
    ~System();
        
    // run system
    // set R|t 
    void init();

    // put next sequences
    void pushNextSequences(const vector<cv::Mat> &im);

    // get results
    void getDepthSequences(vector<cv::Mat> &depth);
    
    // set parameters
    // SPHORB parameters
    void setSPHORB_ratio(const float ratio);
    // SfM parameters
    void setSfM_maxIter(const int maxIter);
    void setSfM_errorTh(const double errTh);
    void setSfM_percentTh(const double perTh);
    // rotation parameters
    void setRotation_method(DEMO::Pixel rMethod);
    // SphereSweeping parameters
    void setSphereSweeping_blockSize(const int blockSize);
    void setSphereSweeping_minDepth(const double minDepth);
    void setSphereSweeping_maxDepth(const double maxDepth);
    void setSphereSweeping_interval(const double interval);
    void setSphereSweeping_numDepth(const int numDepth);
    void setSphereSweeping_method(const DEMO::Pixel ssMethod);
    void decomposeRotationMat(cv::Mat R1);

    vector<int*> allMatch(vector<cv::KeyPoint>* kPoints, Matches* matches);
    cv::Mat allRot(std::vector<cv::Mat> R, vector<cv::KeyPoint>* kPoints, Matches* matches, std::vector<int*> all_match);
    std::vector<cv::Mat> allT(std::vector<cv::Mat> R, std::vector<cv::Mat> t, vector<cv::KeyPoint>* kPoints, Matches* matches);

private:
    void swapT();
    void swapCam();

private:
    // classes
    SphereSweep* mSphsweep;

private:
    // input matrices
    vector<cv::Mat> im, im1K;
    // output 
    vector<cv::Mat> vDepth;
    cv::Mat imDepth;
    
    // image information
    int nCam, nFrame;
    int height, width;

    // 
    int idx;

    // R|t informations
    vector<cv::Mat> R;
    vector<cv::Mat> t;

    // parameters
    // SPHORB
    float ratio;
    // SfM
    int maxIter;
    double errTh;
    double perTh;
    // Rotation
    DEMO::Pixel rMethod;
    // SphereSweeping
    int blockSize;
    double minDepth;
    double maxDepth;
    int numDepth;
    double interval;
    DEMO::Pixel ssMethod;
    cv::Mat confidence;
    bool rigid;
};

} //namespace DEMO

#endif // SYSTEM_H
