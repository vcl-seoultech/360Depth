#include "SfM.hpp"

using namespace std;

namespace DEMO
{

SfM::SfM(const vector<Cartesian> &xyz1, const vector<Cartesian> &xyz2)
{
    this->X1.resize((int)(xyz1.size()));
    this->X2.resize((int)(xyz2.size()));
    copy( xyz1.begin(), xyz1.end(), X1.begin() );
    copy( xyz2.begin(), xyz2.end(), X2.begin() );

    this->Ftmp = cv::Mat::zeros(3, 3, CV_64FC1);
    this->Fmat = cv::Mat::zeros(3, 3, CV_64FC1);
    this->n_total = (int)xyz1.size();
    this->n_inlier = 0;
}

void SfM::run(int max_iter, double threshold, double percent)
{
    this->ransac(max_iter, threshold, percent); // iteration, error threshold, inlier threshold(%)
	this->vote();
	this->decomposeRotationMat();
}

void SfM::ransac(int max_iter, double threshold, double percent)
{
    int c_inliers = 0;
    int cnt = 0;
    double p = .0;
    bool flag = true;
    cv::Mat A = cv::Mat::zeros(8, 9, CV_64FC1);
    cv::Mat U,S,V;
    cv::Mat diagS = cv::Mat::zeros(3, 3, CV_64FC1);
    
    srand( time(NULL) ); // initialize the random seed
    for(int iter = 0; iter < max_iter; iter++)
    {
        for(int i = 0; i < 8; i++)
        {
            r_idx[i] = -1;
        }
        cnt = 0;
        // choose 8 indexes randomly & push them to array
        while(cnt < 8)
        {
            flag = true;
            int randIdx = rand() % n_total;
            for(int j = 0; j < cnt; j++)
            {
                if(r_idx[j] == randIdx)
                {
                    flag = false;
                    break;
                }
            }
            if(flag) 
            {
                r_idx[cnt] = randIdx;
                // [x'*x, x'*y, x'*z, y'*x, y'*y, y'*z, z'*x, z'*y, z'*z]
                //double denom = 1.0/(X1[randIdx].z * X2[randIdx].z);
                
                A.at<double>(cnt, 0) = X1[randIdx].x * X2[randIdx].x;// * denom;
                A.at<double>(cnt, 1) = X1[randIdx].x * X2[randIdx].y;// * denom;
                A.at<double>(cnt, 2) = X1[randIdx].x * X2[randIdx].z;// * denom;
                A.at<double>(cnt, 3) = X1[randIdx].y * X2[randIdx].x;// * denom;
                A.at<double>(cnt, 4) = X1[randIdx].y * X2[randIdx].y;// * denom;
                A.at<double>(cnt, 5) = X1[randIdx].y * X2[randIdx].z;// * denom;
                A.at<double>(cnt, 6) = X1[randIdx].z * X2[randIdx].x;// * denom;
                A.at<double>(cnt, 7) = X1[randIdx].z * X2[randIdx].y;// * denom;
                A.at<double>(cnt, 8) = X1[randIdx].z * X2[randIdx].z;// * denom;
                
                cnt++;
            }
        }
        //for(int i = 0; i < cnt; i++) cout << r_idx[i] << " ";
        //cout << endl;

        /* solve SVD */
        cv::SVDecomp(A, S, U, V, cv::SVD::FULL_UV);
        int idx = 0;
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                Ftmp.at<double>(i, j) =  V.at<double>(8, idx);
                idx++;
            }
        }

        S.release(); U.release(); V.release();
        cv::SVDecomp(Ftmp, S, U, V, cv::SVD::FULL_UV);
        diagS.at<double>(0, 0) = S.at<double>(0);
        diagS.at<double>(1, 1) =     S.at<double>(1);
        
        Ftmp = U * (diagS * V);
        double f22 = Ftmp.at<double>(2, 2);
        Ftmp = Ftmp/f22;

        // evaluate 
        c_inliers = evaluate(threshold);
        if(c_inliers > n_inlier)
        {
            n_inlier = c_inliers;
            //Fmat.release();
            Ftmp.copyTo(Fmat);
            //cout << Fmat << endl;
        } 

        S.release(); U.release(); V.release();


        // exit if n_inlier > required_inliers
        p = (double)(n_inlier)/n_total * 100.0;
        if(p >= percent) break;
    }
    
    cv::decomposeEssentialMat(Fmat, R1, R2, t);
	//cout << "R1: " << R1 << "R2: " << R2 << "t: " << t << endl;
    //cout <<"R.type(): "<<R1.type()<<endl;
    cout << "Ransac results: " << \
    p << "% [" << n_inlier << "/" << n_total <<"]" << endl;
}

int SfM::evaluate(double threshold)
{
    int inlier = 0;
    double err_sum = .0;
    double err = .0;
    //cout << "Threshold: " << threshold << endl;
    for(int i = 0; i < n_total; i++)
    {
        err = this->error(i);
        err_sum += err;
        if(err < threshold) inlier++;
    }
    //cout << "Mean error: " << err_sum/(double)n_total << endl;
    return inlier;
}

double SfM::error(int idx)
{
    /* Structure from Motion using full spherical panoramic cameras, Alain Pagani */

    double err = 0.0;
    
    double data1[3] = {X1[idx].x, X1[idx].y, X1[idx].z};
    double data2[3] = {X2[idx].x, X2[idx].y, X2[idx].z};
    cv::Mat x1(3, 1, CV_64FC1, data1);
    cv::Mat x2(3, 1, CV_64FC1, data2);
    
    cv::Mat Fx1 = Ftmp*x1;
    cv::Mat Fx2 = Ftmp*x2; 
    
    /* Sampson distance error(epsilon s) */
    double denom = pow(Fx1.at<double>(0), 2.0) + pow(Fx1.at<double>(1), 2.0) \
        + pow(Fx2.at<double>(0), 2.0) + pow(Fx2.at<double>(1), 2.0);
    cv::Mat numer = x1.t() * (Fx2);
    err = pow(numer.at<double>(0), 2.0) / denom;


    /* (epsilon p) */
/*
    cv::Mat numer = x1.t() * (Fx2);
    double denom = static_cast<double>( cv::norm(x1.t()) * cv::norm(Fx2) );
    err = abs(numer.at<double>(0))/denom;
*/
    //cout<<"err:"<<err<<", denom:"<<denom << endl;
    return err;
}

void SfM::vote()
{
    // [R1,t+], [R1,t-], [R2,t+], [R2,t-]
    int cnt[4] = {};
    double tmp = .0;
    cv::Mat x1 = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat x2 = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat x2rot1, x2rot2, n, n2, tmp1, tmp2, c1, c2;
    cv::Mat t_m = -1*t;
    for(int i = 0; i < n_total; i++)
    {
        x1.at<double>(0) = X1[i].x;
        x1.at<double>(1) = X1[i].y;
        x1.at<double>(2) = X1[i].z;

        x2.at<double>(0) = X2[i].x;
        x2.at<double>(1) = X2[i].y;
        x2.at<double>(2) = X2[i].z;

        x2rot1 = R1*x2;
        x2rot2 = R2*x2;
        
        // R1
        n = x1.cross(x2rot1);
        n2 = x2rot1.cross(n);

        // t+
        tmp1 = t.t()*n2;
        tmp2 = x1.t()*n2;
        tmp = (tmp1.at<double>(0)) / (tmp2.at<double>(0));
        c1 = tmp*x1;
        
        tmp1 = t_m.t()*n;
        tmp2 = x2rot1.t()*n;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c2 = tmp*x2rot1;
        c2.at<double>(0) = c2.at<double>(0) + t_m.at<double>(0);
        if(x2rot1.at<double>(0) > 0 && c1.at<double>(0) > 0 && c2.at<double>(0) > 0) cnt[0]++;
        else if(x2rot1.at<double>(0) < 0 && c1.at<double>(0) < 0 && c2.at<double>(0) < 0) cnt[0]++;
        
        // t-
        tmp1 = t_m.t()*n2;
        tmp2 = x1.t()*n2;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c1 = tmp*x1;

        tmp1 = t.t()*n;
        tmp2 = x2rot1.t()*n;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c2 = tmp*x2rot1;
        c2.at<double>(0) = c2.at<double>(0) + t.at<double>(0);
        if(x2rot1.at<double>(0) > 0 && c1.at<double>(0) > 0 && c2.at<double>(0) > 0) cnt[1]++;
        else if(x2rot1.at<double>(0) < 0 && c1.at<double>(0) < 0 && c2.at<double>(0) < 0) cnt[1]++;

        // R2
        n = x1.cross(x2rot2);
        n2 = x2rot2.cross(n);
        // t+
        tmp1 = t.t()*n2;
        tmp2 = x1.t()*n2;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c1 = tmp*x1;

        tmp1 = t_m.t()*n;
        tmp2 = x2rot2.t()*n;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c2 = tmp*x2rot2;
        c2.at<double>(0) = c2.at<double>(0) + t_m.at<double>(0);
        if(x2rot2.at<double>(0) > 0 && c1.at<double>(0) > 0 && c2.at<double>(0) > 0) cnt[2]++;
        else if(x2rot2.at<double>(0) < 0 && c1.at<double>(0) < 0 && c2.at<double>(0) < 0) cnt[2]++;
        
        // t-
        tmp1 = t_m.t()*n2;
        tmp2 = x1.t()*n2;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c1 = tmp*x1;

        tmp1 = t.t()*n;
        tmp2 = x2rot2.t()*n;
        tmp = tmp1.at<double>(0) / tmp2.at<double>(0);
        c2 = tmp*x2rot2;
        c2.at<double>(0) = c2.at<double>(0) + t.at<double>(0);
        if(x2rot2.at<double>(0) > 0 && c1.at<double>(0) > 0 && c2.at<double>(0) > 0) cnt[3]++;
        else if(x2rot2.at<double>(0) < 0 && c1.at<double>(0) < 0 && c2.at<double>(0) < 0) cnt[3]++;
    }
    x1.release();
    x2.release();
    int max = -1;
    int idx = -1;
    for(int i = 0; i < 4; i++)
    {
        if(cnt[i] > max)
        {
            max = cnt[i];
            idx = i;
        } 
    }
    cout << "Winner: ";
    if(idx < 2) 
    {
        R1.copyTo(Rmat);
        cout << "R1 & ";
    }
    else 
    {
        R2.copyTo(Rmat);
        cout << "R2 & ";
    }
    if(idx == 0 || idx == 2)
    {
        t.copyTo(Tvec);
        cout << "+t" << endl;
    }
    else 
    {
        t_m.copyTo(Tvec);
        cout << "-t" << endl;
    }
}

void SfM::decomposeRotationMat()
{
    double x1 = atan2(R1.at<double>(2,1), R1.at<double>(2,2));
    double y1 = atan2( -1.0 * R1.at<double>(2,0), \
        sqrt(pow(R1.at<double>(2,1), 2.0)+pow(R1.at<double>(2,2), 2.0)));
    double z1 = atan2(R1.at<double>(1,0), R1.at<double>(0,0));

    double x2 = atan2(R2.at<double>(2,1), R2.at<double>(2,2));
    double y2 = atan2( -1.0 * R2.at<double>(2,0), \
        sqrt(pow(R2.at<double>(2,1), 2.0)+pow(R2.at<double>(2,2), 2.0)));
    double z2 = atan2(R2.at<double>(1,0), R2.at<double>(0,0));
    cout << "x1: " << x1 * (180.0 / M_PI) << 
    "\ty1: " << y1 * (180.0 / M_PI) << \
    "\tz1: " << z1 * (180.0 / M_PI) << endl << \
    "x2: " << x2 * (180.0 / M_PI) << \
    "\ty2: " << y2 * (180.0 / M_PI) << \
    "\tz2: " << z2 * (180.0 / M_PI) << endl;
}

void SfM::getFundamentalMat(cv::Mat &mat)
{
    Fmat.copyTo(mat);
}

void SfM::getRotationMat(cv::Mat &mat)
{
    Rmat.copyTo(mat);
}

void SfM::getTranslationVec(cv::Mat &mat)
{
    Tvec.copyTo(mat);
}

} // namespace DEMO
