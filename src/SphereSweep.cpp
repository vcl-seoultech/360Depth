#include "SphereSweep.hpp"

using namespace std;

namespace DEMO
{

SphereSweep::SphereSweep()
{
    // parameters
    blockSize = 5;
    halfSize = (blockSize-1)/2;
    minDepth = 1.0;
    maxDepth = 10.0;
    interval = 0.2;
    method = DEMO::Pixel::CLOSEST;
    numDepth = (int)((maxDepth-minDepth)/interval) + 1;
}

SphereSweep::SphereSweep(double min_Depth, double max_Depth, double _interval)
{
    // parameters
    blockSize = 7;
    halfSize = (blockSize-1)/2;
    minDepth = min_Depth;
    maxDepth = max_Depth;
    interval = _interval;
    method = DEMO::Pixel::CLOSEST;
    numDepth = (int)((maxDepth-minDepth)/interval) + 1;
}

SphereSweep::SphereSweep(const std::vector<cv::Mat> &im, const std::vector<cv::Mat> &t)
{
    assert(im.size() == t.size());
    numImages = (int)im.size();

    this->im.assign(im.begin(), im.end());
    this->t.assign(t.begin(), t.end());

    imHeight = this->im[0].rows;
    imWidth = this->im[0].cols;
    mHeight = imHeight-1;
    mWidth = imWidth-1;

    // parameters
    blockSize = 5;
    halfSize = (blockSize-1)/2;
    minDepth = 2.0;
    maxDepth = 30.0;
    interval = 0.5;
    method = DEMO::Pixel::CLOSEST;
    numDepth = (int)((maxDepth-minDepth)/interval) + 1;

    // memory allocation
    winDepth = new double[imHeight*imWidth];
    for(auto i = 0; i < imHeight*imWidth; ++i) winDepth[i] = .0;

    cost = new double**[imHeight];
    for (int i = 0; i < imHeight; ++i)
    {
        cost[i] = new double*[imWidth];
        for (int j = 0; j < imWidth; ++j)
        {
            cost[i][j] = new double[numDepth];
            for(int k = 0; k < numDepth; ++k) cost[i][j][k] = .0;
        }
    }

    imDepth = cv::Mat::zeros(this->imHeight, this->imWidth, CV_64FC1);    
}

SphereSweep::~SphereSweep()
{
    delete [] winDepth;
    for (int i = 0; i < imHeight; ++i)
    {
        for (int j = 0; j < imWidth; ++j) delete[] cost[i][j];
        delete[] cost[i];
    }
    delete[] cost;

    imDepth.release();
}

void SphereSweep::run(cv::Mat &dst)
{
    //preprocess    
    //padImages();
    for(int i = 0; i < numImages; ++i)
    {
        //cout<<t[i].at<double>(0)<<"\t"<<t[i].at<double>(1)<<"\t"<<t[i].at<double>(2)<<endl;
        //cv::imwrite(cv::format("test%d.png", i), this->im[i]);    
    }

    cout<<"computing cost ..."<<endl;
    computeCost(Pixel::CLOSEST);
    cout<<"computing winner ..."<<endl;
    computeWinner();

    //cout<<winDepth.rows<<","<<winDepth.cols<<endl;
    //cv::imwrite("test.png", this->winDepth);
    imDepth.copyTo(dst); 

    // set zeros
    for(auto i = 0; i < imHeight*imWidth; ++i) winDepth[i] = .0;
    for (int i = 0; i < imHeight; ++i)
    {
        for (int j = 0; j < imWidth; ++j)
        {
            for(int k = 0; k < numDepth; ++k) cost[i][j][k] = .0;
        }
    }
}

void SphereSweep::setT(const std::vector<cv::Mat> &t)
{
    this->t.clear();
    this->t.resize(t.size());
    this->t.assign(t.begin(), t.end());
}

void SphereSweep::setFrame(const std::vector<cv::Mat> &im)
{
    numImages = (int)im.size();

    this->im.clear();
    this->im.resize(im.size());
    this->im.assign(im.begin(), im.end());

    imHeight = this->im[0].rows;
    imWidth = this->im[0].cols;
    mHeight = imHeight-1;
    mWidth = imWidth-1;

    // memory allocation
    winDepth = new double[imHeight*imWidth];
    for(auto i = 0; i < imHeight*imWidth; ++i) winDepth[i] = .0;

    cost = new double**[imHeight];
    for (int i = 0; i < imHeight; ++i)
    {
        cost[i] = new double*[imWidth];
        for (int j = 0; j < imWidth; ++j)
        {
            cost[i][j] = new double[numDepth];
            for(int k = 0; k < numDepth; ++k) cost[i][j][k] = .0;
        }
    }

    imDepth = cv::Mat::zeros(this->imHeight, this->imWidth, CV_64FC1);  
}

void SphereSweep::nextFrame(const std::vector<cv::Mat> &im)
{
    this->im.clear();
    this->im.resize(im.size());
    assert(this->im.size() != im.size());
    this->im.assign(im.begin(), im.end());
}

void SphereSweep::computeCost(DEMO::Pixel method)
{
    uchar* src_data = im[0].data;
    //cv::Mat debug;
    //im[1].copyTo(debug);
    double dHeight = (double)mHeight, dWidth = (double)mWidth;
    double theta, phi, x, y, z, xy;
    double row, col;
    int ir, ic, nvd = 0;
    for(auto idx = 1; idx < numImages; ++idx)
    {
        uchar* data = im[idx].data;
        double t0 = t[idx].at<double>(0);
        double t1 = t[idx].at<double>(1);
        double t2 = t[idx].at<double>(2);
        
        for(double vd = minDepth; vd <= maxDepth; vd += interval)
        { 
            // get pixels
            for(double r = .0; r < dHeight; r += 1.0)
            {
                ir = (int)r;
                dPair pt;
                for(double c = .0; c < dWidth; c+= 1.0)
                {
                    ic = (int)c;
                    // rho = vd
                    theta = r/dHeight * M_PI;
                    phi = c/dWidth * (2.0*M_PI);

                    // Spherical to Cartesian
                    x = vd * sin(theta)*cos(phi);
                    y = vd * sin(theta)*sin(phi);
                    z = vd * cos(theta);

                    // compute phi, theta from t[idx] to xyz
                    x -= t0;
                    y -= t1;
                    z -= t2;

                    // Cartesian to Spherical
                    xy = sqrt( pow(x, 2.0) + pow(y, 2.0) );
                    phi = atan2(y, x);
                    if(y < .0) phi += (2.0*M_PI);
                    theta = atan2(xy, z);

                    row = (theta/M_PI) * dHeight;
                    col = (phi/(2.0*M_PI)) * dWidth;

                    // get Colors
                    //cout<<row<<","<<col<<endl;
                    pt = make_pair(row, col);
                    unsigned char bgr[3];
                    if(method == DEMO::Pixel::MEAN)
                    {
                        int r0i = (int)row;
                        int c0i = (int)col;

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
                        for(int k = 0; k < 4; ++k)
                        {
                            double d = calcDistance(p[k], pt);
                            if(d == 0.0) w[k] = INF;
                            else w[k] = 1.0/d;
                            wsum += w[k];
                        }
                        for(int k = 0; k < 4; ++k) w[k] = w[k]/wsum;

                        for(int k = 0; k < 3; ++k) bgr[k] = 0;
                        for(int k = 0; k < 4; ++k)
                        {
                            int irow = (int)p[k].first;
                            int icol = (int)p[k].second;
                            if(icol >= imWidth -1) icol -= imWidth;
                            if(irow >= imHeight -1){ irow--; icol = (icol+imWidth/2)%imWidth; }
                            bgr[0] += (unsigned char)(w[k]*(double)data[(irow*imWidth + icol)*3]);
                            bgr[1] += (unsigned char)(w[k]*(double)data[(irow*imWidth + icol)*3 + 1]);
                            bgr[2] += (unsigned char)(w[k]*(double)data[(irow*imWidth + icol)*3 + 2]);
                        }
                    }
                    else if(method == DEMO::Pixel::CLOSEST)
                    {
                        int irow = round(pt.first);
                        int icol = round(pt.second);

                        bgr[0] = data[(irow*imWidth + icol)*3];
                        bgr[1] = data[(irow*imWidth + icol)*3 + 1];
                        bgr[2] = data[(irow*imWidth + icol)*3 + 2];
                    }
                    cost[ir][ic][nvd] += abs(bgr[0] - src_data[(ir*imWidth + ic)*3])/3.0;
                    cost[ir][ic][nvd] += abs(bgr[1] - src_data[(ir*imWidth + ic)*3 + 1])/3.0;
                    cost[ir][ic][nvd] += abs(bgr[2] - src_data[(ir*imWidth + ic)*3 + 2])/3.0;
                }
            }
            //cout<<nvd<<endl;
            ++nvd;   
        }
        nvd = 0;
    }
}

void SphereSweep::computeWinner()
{
    for(int r = halfSize; r < imHeight-halfSize; ++r)
    {
        for(int c = halfSize; c < imWidth-halfSize; ++c)
        {
            double min = 1e9;
            for(int vd = 0; vd < numDepth; ++vd)
            {
                double cost_sum = .0;
                for(int i = r-halfSize; i <= r+halfSize; ++i)
                {
                    for(int j = c-halfSize; j <= c+halfSize; ++j) cost_sum += cost[i][j][vd];
                }
                if(cost_sum < min)
                {
                    min = cost_sum;
                    winDepth[r*imWidth + c] = minDepth + interval*vd;
                }
            }
            //cout<<r<<" "<<c<<endl;
        }
    }

    cv::Mat tmp;
    cout<<"start copying ..."<<endl;
    for(int r = 0; r < imHeight; ++r)
    {
        double* ptrImDepth = this->imDepth.ptr<double>(r);
        for(int c = 0; c < imWidth; ++c)
        {
            ptrImDepth[c] = winDepth[r*imWidth + c];
            //cout<<r<<" "<<c<<endl;
        } 
    }
    cout<<"Done!"<<endl;
}

double SphereSweep::computeWinner(const int r, const int c)
{
    double winDepth = -1.0;
    int cost, minCost = 1e9;
    double virtualDepth = minDepth;
    while(virtualDepth <= maxDepth)
    {
        // compute cost for each depth
        cost = computeCost(r, c, virtualDepth); // these r & c are r+halfSize & c+halfSize
        if(cost < minCost)
        {
            minCost = cost;
            winDepth = virtualDepth;
        }
        virtualDepth += interval;
    }
    //cout<<winDepth<<endl;
    return winDepth;
}

int SphereSweep::computeCost(int r, int c, double virtualDepth)
{
    // make mask 
    cv::Mat mask(1, blockSize*blockSize, CV_8UC3);
    uchar *mask_data = mask.data;
    uchar *src_data = im[0].data;
    int sr = -1*halfSize;
    int sc = -1*halfSize;
    for(auto i = 0; i < blockSize*blockSize; i++)
    {
        mask_data[i*3] = src_data[( (r+sr)*padWidth + (c+sc) )*3];
        mask_data[i*3 + 1] = src_data[( (r+sr)*padWidth + (c+sc) )*3 + 1];
        mask_data[i*3 + 2] = src_data[( (r+sr)*padWidth + (c+sc) )*3 + 2];
        sc++;
        if(sc > halfSize)
        {
            sr++;
            sc = -1*halfSize;
        }
    }
    
    int costsum = 0, cost = 0;
    cv::Mat color(1,1,CV_8UC3);
    uchar *color_data = color.data;
    for(auto idx = 1; idx < numImages; idx++)
    {
        /* getColorfromIdx */
        sr = -1*halfSize;
        sc = -1*halfSize;
        for(auto i = 0; i < blockSize*blockSize; i++)
        {
            dPair pt = getPixel(r+sr, c+sc, virtualDepth, idx);
            getColor(color, pt, idx, DEMO::Pixel::CLOSEST);
        
            //cout<<(int)color_data[0] <<","<<(int)color_data[1]<<","<<(int)color_data[2]<<endl;
            for(auto j = 0; j < 3; j++) cost += (int)abs(color_data[j] - mask_data[i*3 + j]);

            sc++;
            if(sc > halfSize)
            {
                sr++;
                sc = -1*halfSize;
            }
        }
        costsum += cost;
    }
    return costsum;
}

dPair SphereSweep::getPixel(int r, int c, double virtualDepth, int idx)
{
    /* compute pixel from dst */
    // convert Spherical Coordinate to cartesian coordinate xyz
    Spherical sph;
    sph.rho = virtualDepth;
    sph.theta = (double)(r - halfSize)/(double)mHeight * M_PI;
    sph.phi = (double)(c - halfSize)/(double)mWidth * (2.0*M_PI);  
    Cartesian xyz = SphericalToCartesian(sph);

    // compute phi, theta from t[idx] to xyz
    xyz.x -= t[idx].at<double>(0);
    xyz.y -= t[idx].at<double>(1);
    xyz.z -= t[idx].at<double>(2);

    sph = CartesianToSpherical(xyz);
    double row = (sph.theta/M_PI * mHeight) + (double)halfSize;
    double col = (sph.phi/(2.0*M_PI) * mWidth) + (double)halfSize;

    dPair pt = make_pair(row, col);

    // for debugging
    double sInterval = 200.0/maxDepth;
    int irow = (int)round(row), icol = (int)round(col);
    int green = 200 - (int)(virtualDepth*sInterval);
    int red = (int)(virtualDepth*sInterval) + 55;
    //cout<<virtualDepth<<", red: "<<red<<endl;
    //cv::circle(im[idx], cv::Point(icol, irow), 1, cv::Scalar(0, green, red), -1);
    im[idx].at<cv::Vec3b>(irow, icol)[0] = 0;
    im[idx].at<cv::Vec3b>(irow, icol)[1] = green;
    im[idx].at<cv::Vec3b>(irow, icol)[2] = red;

    return pt;
}

void SphereSweep::getColor(cv::Mat &dst, dPair pt, int idx, DEMO::Pixel method)
{
    cv::Mat src;
    im[idx].copyTo(src);
    cv::Mat color(1, 1, CV_8UC3);
    uchar* src_data = src.data;
    uchar* color_data = color.data;

    if(method == DEMO::Pixel::MEAN)
    {
        int r0i = (int)pt.first;
        int c0i = (int)pt.second;

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
            double d = calcDistance(p[k], pt);
            if(d == 0.0) w[k] = INF;
            else w[k] = 1.0/d;
            wsum += w[k];
        }
        for(int k = 0; k < 4; k++) w[k] = w[k]/wsum;

        color_data[0] = 0.0;
        color_data[1] = 0.0;
        color_data[2] = 0.0;
        for(int k = 0; k < 4; k++)
        {
            int row = (int)p[k].first;
            int col = (int)p[k].second;
            //if(col >= imWidth -1) col = 0;
            //if(row >= imHeight -1){ row--; col = (col+imWidth/2)%imWidth; }
            color_data[0] += (int)(w[k]*(double)src_data[(row*padWidth + col)*3]);
            color_data[1] += (int)(w[k]*(double)src_data[(row*padWidth + col)*3 + 1]);
            color_data[2] += (int)(w[k]*(double)src_data[(row*padWidth + col)*3 + 2]);
        }
    }
    else if(method == DEMO::Pixel::CLOSEST)
    {
        int row = round(pt.first);
        int col = round(pt.second);
        //if(col >= imWidth -1) col = 0;
        //if(row >= imHeight -1){ row--; col = (col+imWidth/2)%imWidth; }
        color_data[0] = src_data[(row*padWidth + col)*3];
        color_data[1] = src_data[(row*padWidth + col)*3 + 1];
        color_data[2] = src_data[(row*padWidth + col)*3 + 2];
    }
    color.copyTo(dst);
}

void SphereSweep::padImages()
{
    padHeight = imHeight+halfSize*2;
    padWidth = imWidth+halfSize*2;
    for(auto i = 0; i < numImages; ++i)
    {
        cv::Mat padIm(padHeight, padWidth, CV_8UC3);
        uchar *pad_data = padIm.data;
        uchar *src_data = im[i].data;

        // copy original image to padimage
        for(int r = 0; r < imHeight; ++r)
        {
            for(int c = 0; c < imWidth; ++c)
            {
                pad_data[((r+halfSize)*padWidth + (c+halfSize))*3] = src_data[(r*imWidth + c)*3];
                pad_data[((r+halfSize)*padWidth + (c+halfSize))*3 + 1] = src_data[(r*imWidth + c)*3 + 1];
                pad_data[((r+halfSize)*padWidth + (c+halfSize))*3 + 2] = src_data[(r*imWidth + c)*3 + 2];
            }
        }

        //cv::GaussianBlur(padIm, padIm, cv::Size(blockSize, blockSize), 0, 0);
        /* pad Images */
        // pad up


        // pad down
        // pad left
        // pad right

        padIm.copyTo(im[i]);
    }
}

void SphereSweep::setBlockSize(const int blockSize)
{
    if(blockSize%2 == 0)
    {
        cout<<"BlockSize must be odd"<<endl;
        return;
    }
    else
    {
        this->blockSize = blockSize;
        halfSize = (this->blockSize-1)/2;
    } 
}
void SphereSweep::setMinDepth(const double minDepth)
{
    this->minDepth = minDepth;
}
void SphereSweep::setMaxDepth(const double maxDepth)
{
    this->maxDepth = maxDepth;
}
void SphereSweep::setInterval(const double interval)
{
    this->interval = interval;
    this->numDepth = (int)((this->maxDepth - this->minDepth)/this->interval) + 1;
}
void SphereSweep::setNumDepth(const int numDepth)
{
    this->numDepth = numDepth;
    this->interval = (this->maxDepth - this->minDepth)/(double)this->numDepth;
}
void SphereSweep::setMethod(const DEMO::Pixel method)
{
    this->method = method;
}

} //namespace DEMO
