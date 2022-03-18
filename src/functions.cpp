#include "functions.hpp"

namespace DEMO
{

void rotateImg(const cv::Mat &src, cv::Mat &dst, const cv::Mat &Rmat)
{
    int height = src.rows;
    int width = src.cols;
    dst.create((src.size()), src.type());
    cv::Mat R = Rmat.t();
    
    
    // preprocess
    Coordinate *coord = new Coordinate(height, width);
    coord->ImgToCartesian(false);
    coord->rotateCartesian(R, false);
    coord->CartesianToImg();
    cv::Mat pixel;
    coord->imgm.copyTo(pixel);
    delete coord;
    
    if(src.type() == CV_8UC1)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        for(int i = 0; i < height*width; i++)
        {
            int row = (int)pixel.at<double>(0, i);
            int col = (int)pixel.at<double>(1, i);
            dst_data[i] = src_data[row*width + col];
        }
    }
    else if(src.type() == CV_8UC3)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        for(int i = 0; i < height*width; i++)
        {
            
            int r0i = (int)pixel.at<double>(0, i);
            int c0i = (int)pixel.at<double>(1, i);
/*
            //[b, g, r]
            dst_data[i*3] = src_data[r0i*width*3 + c0i*3];
            dst_data[i*3 + 1] = src_data[r0i*width*3 + c0i*3 + 1];
            dst_data[i*3 + 2] = src_data[r0i*width*3 + c0i*3 + 2];
*/

            dPair f = make_pair(pixel.at<double>(0, i), pixel.at<double>(1, i)); 
            double r0 = (double)r0i;
            double c0 = (double)c0i;
            double r1 = r0 + 1.0;
            double c1 = c0 + 1.0;
    
            dPair p[4]; //00 01 10 11
            p[0] = make_pair(r0, c0);
            p[1] = make_pair(r0, c1);
            p[2] = make_pair(r1, c0);
            p[3] = make_pair(r1, c1);

            double w[4];
            double wsum = 0.0;
            for(int k = 0; k < 4; k++)
            {
                double d = calcDistance(p[k], f);
                if(d == 0.0) w[k] = INF;
                else w[k] = 1.0/d;
                wsum += w[k];
            }
            for(int k = 0; k < 4; k++) w[k] = w[k]/wsum;
            //cout<<r0i<<","<<c0i<<"\t"<<f.first<<","<<f.second<<
            //"\t["<<r0<<","<<c0<<"\t"<<r1<<","<<c1<<"] ["<<w[0]<<","<<w[1]<<","<<w[2]<<","<<w[3]<<"]"<<endl;

            dst_data[i*3] = 0.0;
            dst_data[i*3 + 1] = 0.0;
            dst_data[i*3 + 2] = 0.0;
            for(int k = 0; k < 4; k++)
            {
                int row = (int)p[k].first;
                int col = (int)p[k].second;
                if(col >= width -1) col = 0;
                if(row >= height -1){ row--; col = (col+width/2)%width; }
                dst_data[i*3] += (int)(w[k]*src_data[row*width*3 + col*3]);
                dst_data[i*3 + 1] +=(int)(w[k]*src_data[row*width*3 + col*3 + 1]);
                dst_data[i*3 + 2] += (int)(w[k]*src_data[row*width*3 + col*3 + 2]);
            }
            
        }
    }
    else cout << "Error occured..." << endl;
}

void rotateImg(const cv::Mat &src, cv::Mat &dst, const cv::Mat &Rmat, DEMO::Pixel method)
{
    int height = src.rows;
    int width = src.cols;
    dst.create((src.size()), src.type());
    cv::Mat R = Rmat.t();
    
    
    // preprocess
    cv::Mat pixel;
    {
        Coordinate *coord = new Coordinate(height, width);
        coord->ImgToCartesian(false);
        coord->rotateCartesian(R, false);
        coord->CartesianToImg();  
        coord->imgm.copyTo(pixel);
        delete coord;
    }
    
    if(src.type() == CV_8UC1)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        if(method == DEMO::Pixel::FIX)
        {
            cout<<"Pixel Method: FIX"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int row = (int)pixel.at<double>(0, i);
                int col = (int)pixel.at<double>(1, i);
                dst_data[i] = src_data[row*width + col];
            }
        }
        else if(method == DEMO::Pixel::MEAN)
        {
            cout<<"Pixel Method: MEAN"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int r0i = (int)pixel.at<double>(0, i);
                int c0i = (int)pixel.at<double>(1, i);

                dPair f = make_pair(pixel.at<double>(0, i), pixel.at<double>(1, i)); 
                double r0 = (double)r0i;
                double c0 = (double)c0i;
                double r1 = r0 + 1.0;
                double c1 = c0 + 1.0;
        
                dPair p[4]; //00 01 10 11
                p[0] = make_pair(r0, c0);
                p[1] = make_pair(r0, c1);
                p[2] = make_pair(r1, c0);
                p[3] = make_pair(r1, c1);

                double w[4];
                double wsum = 0.0;
                for(int k = 0; k < 4; k++)
                {
                    double d = calcDistance(p[k], f);
                    if(d == 0.0) w[k] = INF;
                    else w[k] = 1.0/d;
                    wsum += w[k];
                }
                for(int k = 0; k < 4; k++) w[k] = w[k]/wsum;
            
                dst_data[i*3] = 0.0;
                for(int k = 0; k < 4; k++)
                {
                    int row = (int)p[k].first;
                    int col = (int)p[k].second;
                    if(col >= width) col = 0;
                    if(row >= height) row = height-1;
                    dst_data[i] += (int)(w[k]*(double)src_data[row*width + col]);
                }
            }
        }
        else if(method == DEMO::Pixel::CLOSEST)
        {
            cout<<"Pixel Method: CLOSEST"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int row = round(pixel.at<double>(0, i));
                int col = round(pixel.at<double>(1, i));

                if(col >= width) col = 0;
                if(row >= height) row = height -1;
                dst_data[i] = src_data[row*width + col];
            }
        }
    }
    else if(src.type() == CV_8UC3)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        if(method == DEMO::Pixel::FIX)
        {
            cout<<"Pixel Method: FIX"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int r0i = (int)pixel.at<double>(0, i);
                int c0i = (int)pixel.at<double>(1, i);
                //[b, g, r]
                dst_data[i*3] = src_data[r0i*width*3 + c0i*3];
                dst_data[i*3 + 1] = src_data[r0i*width*3 + c0i*3 + 1];
                dst_data[i*3 + 2] = src_data[r0i*width*3 + c0i*3 + 2];
            }
        }
        else if(method == DEMO::Pixel::MEAN)
        {
            cout<<"Pixel Method: MEAN"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int r0i = (int)pixel.at<double>(0, i);
                int c0i = (int)pixel.at<double>(1, i);

                dPair f = make_pair(pixel.at<double>(0, i), pixel.at<double>(1, i)); 
                double r0 = (double)r0i;
                double c0 = (double)c0i;
                double r1 = r0 + 1.0;
                double c1 = c0 + 1.0;
        
                dPair p[4]; //00 01 10 11
                p[0] = make_pair(r0, c0);
                p[1] = make_pair(r0, c1);
                p[2] = make_pair(r1, c0);
                p[3] = make_pair(r1, c1);

                double w[4];
                double wsum = 0.0;
                for(int k = 0; k < 4; k++)
                {
                    double d = calcDistance(p[k], f);
                    if(d == 0.0) w[k] = INF;
                    else w[k] = 1.0/d;
                    wsum += w[k];
                }
                for(int k = 0; k < 4; k++) w[k] = w[k]/wsum;
            
                dst_data[i*3] = 0.0;
                dst_data[i*3 + 1] = 0.0;
                dst_data[i*3 + 2] = 0.0;
                for(int k = 0; k < 4; k++)
                {
                    int row = (int)p[k].first;
                    int col = (int)p[k].second;
                    if(col >= width) col = 0;
                    if(row >= height) row = height-1;
                    dst_data[i*3] += (int)(w[k]*(double)src_data[row*width*3 + col*3]);
                    dst_data[i*3 + 1] += (int)(w[k]*(double)src_data[row*width*3 + col*3 + 1]);
                    dst_data[i*3 + 2] += (int)(w[k]*(double)src_data[row*width*3 + col*3 + 2]);
                }
            }
        }
        else if(method == DEMO::Pixel::CLOSEST)
        {
            cout<<"Pixel Method: CLOSEST"<<endl;
            for(int i = 0; i < height*width; i++)
            {
                int row = round(pixel.at<double>(0, i));
                int col = round(pixel.at<double>(1, i));

                if(col >= width) col = 0;
                if(row >= height) row = height -1;
                dst_data[i*3] = src_data[row*width*3 + col*3];
                dst_data[i*3 + 1] = src_data[row*width*3 + col*3 + 1];
                dst_data[i*3 + 2] = src_data[row*width*3 + col*3 + 2];
            }
        }
    }
    else cout << "Error occured..." << endl;
}

void rotateImg(const cv::Mat &src, cv::Mat &dst, const double ax, const double ay, const double az)
{
    int height = src.rows;
    int width = src.cols;
    dst.create((src.size()), src.type());    
    
    // preprocess
    cv::Mat Rx, Ry, Rz;
    composeRotationMat(Rx, ax, 0, .0f);
    composeRotationMat(Ry, .0, ay, .0f);
    composeRotationMat(Rz, .0, .0, az);
    Coordinate *coord = new Coordinate(height, width);
    coord->ImgToCartesian(false);
    coord->rotateCartesian(Rz, false);
    coord->rotateCartesian(Ry, false);
    coord->CartesianToImg();
    cv::Mat pixel;
    coord->imgm.copyTo(pixel);
    delete coord;
    
    if(src.type() == CV_8UC1)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        for(int i = 0; i < height*width; i++)
        {
            int row = (int)pixel.at<double>(0, i);
            int col = (int)pixel.at<double>(1, i);
            dst_data[i] = src_data[row*width + col];
        }
    }
    else if(src.type() == CV_8UC3)
    {
        uchar *dst_data = (uchar*)dst.data;
        uchar *src_data = (uchar*)src.data;
        for(int i = 0; i < height*width; i++)
        {
            // NEED TO IMPROVE IMAGE QUALITY!!!
            int row = (int)pixel.at<double>(0, i);
            int col = (int)pixel.at<double>(1, i);
            //[b, g, r]
            dst_data[i*3] = src_data[row*width*3 + col*3];
            dst_data[i*3 + 1] = src_data[row*width*3 + col*3 + 1];
            dst_data[i*3 + 2] = src_data[row*width*3 + col*3 + 2];
        }
    }
    else cout << "Error occured..." << endl;
}

void composeRotationMat(cv::Mat &R, const double x, const double y, const double z)
{
    cv::Mat Rx = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat Ry = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat Rz = cv::Mat::zeros(3, 3, CV_64F);

    Rx.at<double>(0,0) = 1.0;
    Rx.at<double>(1,1) = cos(x);
    Rx.at<double>(1,2) = -1.0*sin(x);
    Rx.at<double>(2,1) = sin(x);
    Rx.at<double>(2,2) = cos(x);

    Ry.at<double>(1,1) = 1.0;
    Ry.at<double>(0,0) = cos(y);
    Ry.at<double>(0,2) = sin(y);
    Ry.at<double>(2,0) = -1.0*sin(y);
    Ry.at<double>(2,2) = cos(y);

    Rz.at<double>(2,2) = 1.0;
    Rz.at<double>(0,0) = cos(z);
    Rz.at<double>(0,1) = -1.0*sin(z);
    Rz.at<double>(1,0) = sin(z);
    Rz.at<double>(1,1) = cos(z);

    cv::Mat tmp = Rx*(Ry*Rz);
    //cv::Mat tmp = Rx*(Ry*Rz);
    tmp.copyTo(R);

    //cout << z * 180.0/M_PI << " " << y * 180.0/M_PI << endl;
}

double calcDistance(dPair a, dPair b)
{
    double dr = pow(abs(a.first - b.first), 2.0);
    double dc = pow(abs(a.second - b.second), 2.0);
    return sqrt(dr+dc);
}

} //namespace DEMO