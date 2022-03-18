#ifndef SFM_H
#define SFM_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "type.hpp"

using namespace std;

namespace DEMO
{

class SfM
{

public:
    SfM(const vector<Cartesian> &xyz1, const vector<Cartesian> &xyz2);

    void run(int max_iter, double threshold, double percent);

    // run ransac
    void ransac(int max_iter, double threshold, double percent);

    // vote correct R and t
    void vote();

    // decompose & print R, t
    void decomposeRotationMat();

    // return matrices
    void getFundamentalMat(cv::Mat &mat);
    void getRotationMat(cv::Mat &mat);
    void getTranslationVec(cv::Mat &mat);

private:
    // return number of inliers
    int evaluate(double threshold);
    // evalulate error
    double error(int idx);

private:
    int n_total; //number of total pts
    int n_inlier; //number of inliers
    int r_idx[8]; // randomly chosen indexes;

    vector<Cartesian> X1, X2;

    cv::Mat Fmat, Ftmp;
    cv::Mat R1, R2, t;
    cv::Mat Rmat, Tvec;

};

} // namespace DEMO

#endif // SFM_H
