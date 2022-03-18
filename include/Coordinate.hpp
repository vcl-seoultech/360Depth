#ifndef COORDINATE_H
#define COORDINATE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "type.hpp"

using namespace std;

namespace DEMO
{

class Coordinate
{
public:
    // start coordinates
    Coordinate(const int r, const int c, const vector<cv::KeyPoint> &kp1, const vector<cv::KeyPoint> &kp2, const vector<cv::DMatch> &matches);
    Coordinate(const int r, const int c);

    // return matched points as <row, col>format
    void get_pts(vector<img_d> &vecd, int idx);

    // convert <row, col>format to <x, y, z>format
    void ImgToCartesian(); // type: Cartesian
    void ImgToCartesian(bool kp); // type: cv::Mat

    // convert <x, y, z>format to <row, col>
    void CartesianToImg();
    // convert <x, y, z>format to <rho, phi, theta>format
    void CartesianToSpherical(vector<Cartesian> &cart, vector<Spherical> &sph);
    
       
    // rotate 3d cartesian pixels(cv::Mat)
    void rotateCartesian(cv::Mat &R, bool kp);    

public:
    vector<img_d> imgd1, imgd2;
    vector<Cartesian> xyz1, xyz2;
    vector<Equi> equi1, equi2;
    vector<Spherical> sph1, sph2;

    cv::Mat cart, cartKp;
    cv::Mat imgm;

private:
    int nFeatures; // number of features;
    int Height, Width;
    int mHeight, mWidth;

};

void SphericalToCartesian(vector<Spherical> &src, vector<Cartesian> &dst);
Cartesian SphericalToCartesian(Spherical src);
Spherical CartesianToSpherical(Cartesian src);

} // namespace DEMO

#endif // COORDINATE_H