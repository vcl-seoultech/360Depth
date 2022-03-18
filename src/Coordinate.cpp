#include "Coordinate.hpp"

using namespace std;

namespace DEMO
{

Coordinate::Coordinate(const int r, const int c, const vector<cv::KeyPoint> &kp1, const vector<cv::KeyPoint> &kp2, const vector<cv::DMatch> &matches)
{
    img_d tmp1, tmp2;
    Height = r;
    Width = c;
    mHeight = Height-1;
    mWidth = Width-1;
    nFeatures = (int)matches.size();
    cout << "Num features: " << nFeatures << endl;
    for(auto i = 0; i < nFeatures; i++)
	{
        tmp1.row = kp1[matches[i].queryIdx].pt.y;
        tmp1.col = kp1[matches[i].queryIdx].pt.x;
        tmp2.row = kp2[matches[i].trainIdx].pt.y;
        tmp2.col = kp2[matches[i].trainIdx].pt.x;
		imgd1.push_back(tmp1);
        imgd2.push_back(tmp2);
	}
/*
    for(auto i = 0; i < nFeatures; i++)
    {
        cout<<"(" << imgd1[i].row << ", " << imgd1[i].col << 
        ") (" << imgd2[i].row << ", " << imgd2[i].col << ")"
        << endl;
    }
*/
}

Coordinate::Coordinate(const int r, const int c)
{
    Height = r;
    Width = c;
    mHeight = Height-1;
    mWidth = Width-1;
}

void Coordinate::get_pts(vector<img_d> &vecd, int idx)
{
    vecd.resize(nFeatures);
    if(idx == 1) copy(imgd1.begin(), imgd1.end(), vecd.begin() );
    else if(idx == 2) copy(imgd2.begin(), imgd2.end(), vecd.begin() );
    else cout << "Usage: get_pts(vector<img_d> storage, idx[1 or 2])" << endl;
}

void Coordinate::ImgToCartesian()
{
    // img_d -> cartesian(xyz)
    double theta, phi;
    Cartesian tmp;
    for(int i = 0; i < nFeatures; i++)
    {
        theta = (imgd1[i].row / (double)mHeight) * M_PI;
        phi = (imgd1[i].col / (double)mWidth) * (2.0*M_PI);
        tmp.x = sin(theta)*cos(phi);
        tmp.y = sin(theta)*sin(phi);
        tmp.z = cos(theta);
        this->xyz1.push_back(tmp);

        theta = (imgd2[i].row / (double)mHeight) * M_PI;
        phi = (imgd2[i].col / (double)mWidth) * (2.0*M_PI);
        tmp.x = sin(theta)*cos(phi);
        tmp.y = sin(theta)*sin(phi);
        tmp.z = cos(theta);
        this->xyz2.push_back(tmp);
    }
    assert(xyz1.size() == nFeatures);
}

void Coordinate::ImgToCartesian(bool kp)
{
    if(kp) // keyPoint or not
    {
        cartKp.create(3, nFeatures, CV_64FC1);
        // will be updated
    }
    else // if not keypoint generate cartesian points for all pixels with cvMat type
    {
        int n_pixel = Height * Width;
        cart.create(3, n_pixel, CV_64FC1);
        /*  [x0]   [xn]
            [y0]...[yn]
            [z0]   [zn] */

        double theta, phi;
        for(int r = 0; r < Height; r++)
        {
            for(int c = 0; c < Width; c++)
            {
                theta = (double)r/(double)mHeight* M_PI;
                phi = (double)c/(double)mWidth * (2.0*M_PI);

                cart.at<double>(0,r*Width + c) = (sin(theta)*cos(phi));
                cart.at<double>(1,r*Width + c) = (sin(theta)*sin(phi));
                cart.at<double>(2,r*Width + c) = cos(theta);
                
                /*cout << cart.at<double>(0, r*width + c) * (180.0/M_PI) << \
                cart.at<double>(1, r*width + c) * (180.0/M_PI) << \
                cart.at<double>(2, r*width + c) * (180.0/M_PI) << endl;*/
            }
        }
    }
}

void Coordinate::CartesianToImg()
{
    double x, y, z, xy, theta, phi;
    imgm.create(2, Height*Width, CV_64FC1);
    /*  [r0]...[rn]
        [c0]   [cn] */
    for(int i = 0; i < Height*Width; i++)
    {
        x = cart.at<double>(0,i);
        y = cart.at<double>(1,i);
        z = cart.at<double>(2,i);
        xy = pow(x, 2.0) + pow(y, 2.0);
        phi = atan2(y, x);
        if(y < 0) phi += (2.0*M_PI);
        theta = atan2(sqrt(xy), z);
        imgm.at<double>(0,i) = theta/M_PI * (double)mHeight; //row
        imgm.at<double>(1,i) = phi/(2.0*M_PI) * (double)mWidth; //column
        
        //cout << imgm.at<double>(0, i) << "," << imgm.at<double>(1,i) << endl;
    }
}

void Coordinate::CartesianToSpherical(vector<Cartesian> &cart, vector<Spherical> &sph)
{
    int n = cart.size();
    double x, y, z;
    Spherical tmp;
    sph.clear();
    for(int i = 0; i < n; i++)
    {
        x = round(10.0*cart[i].x)*0.1;
        y = round(10.0*cart[i].y)*0.1;
        z = round(10.0*cart[i].z)*0.1;

        double xy = (pow(x, 2.0) + pow(y, 2.0));
        xy = sqrt(xy);
        tmp.rho = sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
        tmp.phi = atan2(y, x);
        if(y < 0) tmp.phi += (2.0*M_PI);
        tmp.theta = atan2(xy, z);
        sph.push_back(tmp);
    }
    //cout << "xyz: " << x << " " << y << " " << z << endl;
}


void Coordinate::rotateCartesian(cv::Mat &R, bool kp)
{
    if(kp) cartKp = R * cartKp;
    else cart = R * cart;
}

void SphericalToCartesian(vector<Spherical> &src, vector<Cartesian> &dst)
{
    dst.clear();
    Cartesian tmp;
    for(auto i = 0; i < src.size(); i++)
    {
        tmp.x = src[i].rho * sin(src[i].theta)*cos(src[i].phi);
        tmp.y = src[i].rho * sin(src[i].theta)*sin(src[i].phi);
        tmp.z = src[i].rho * cos(src[i].theta);
        dst.push_back(tmp);
    }
}

Cartesian SphericalToCartesian(Spherical src)
{
    Cartesian dst;

    dst.x = src.rho * sin(src.theta)*cos(src.phi);
    dst.y = src.rho * sin(src.theta)*sin(src.phi);
    dst.z = src.rho * cos(src.theta);

    return dst;
}

Spherical CartesianToSpherical(Cartesian src)
{
    Spherical dst;

    double xy = (pow(src.x, 2.0) + pow(src.y, 2.0));
    xy = sqrt(xy);
    dst.rho = sqrt( pow(src.x, 2.0)+pow(src.y, 2.0)+pow(src.z, 2.0) );
    dst.phi = atan2(src.y, src.x);
    if(src.y < .0) dst.phi += (2.0*M_PI);
    dst.theta = atan2(xy, src.z);

    return dst;
}

}; // namespace DEMO
