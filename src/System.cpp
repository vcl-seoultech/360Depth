#include "System.hpp"

using namespace std;

namespace DEMO
{//

System::System(const vector<cv::Mat> &im, int idx, double max_Depth, double _interval, bool _rigid):nFrame(0)
{
    assert( (im[0].cols == im[0].cols) && \
            (im[0].rows == im[0].rows) && \
            (im[0].cols != im[0].rows * 2) );

	nCam = (int)im.size();
	this->im.resize(nCam);
	this->im1K.resize(nCam);
	copy(im.begin(), im.end(), this->im.begin());
	copy(im.begin(), im.end(), this->im1K.begin());

	this->idx = idx;
	
	for(int i = 0; i < nCam; ++i)
	{
		cv::resize(this->im1K[i], this->im1K[i], cv::Size(1280, 640), 0, 0, CV_INTER_NN);
	}

	height = this->im[0].rows;
	width = this->im[0].cols;

	{
		double rdata[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
		cv::Mat R0(3,3,CV_64FC1, rdata);
		R.push_back(R0);

		cv::Mat t0 = cv::Mat::zeros(3,1,CV_64FC1);
		t.push_back(t0);
	}

	//mSphsweep = new SphereSweep();

	// set parameters
	// SPHORB
	ratio = 0.75f;
	// SfM
	maxIter = 10000;
	errTh = 0.0001;
	perTh = 99.0;
	// Rotation
	rMethod = DEMO::Pixel::CLOSEST;
	// SphereSweeping
	blockSize = 5;
    //minDepth = 1.0;
    minDepth = 0.0;
    maxDepth = max_Depth;
    interval = _interval;

	mSphsweep = new SphereSweep(minDepth, maxDepth, interval);
    //mSphsweep = new SphereSweep();
	numDepth = (int)((maxDepth-minDepth)/interval) + 1;
    ssMethod = DEMO::Pixel::MEAN;

	rigid = _rigid;
	if (rigid)
	{
	confidence = cv::Mat::ones(im[0].size(), CV_64FC1);
	for (int i = 0; i < confidence.cols; i++)
	{
		for (int j = 0; j < confidence.rows; j++)
		{
			confidence.at<double>(j, i) = sin(j / static_cast<double>(confidence.rows) * M_PI) * 255.0;
			//confidence.at<double>(j, i) = 0.;
		}
	}
	confidence.convertTo(confidence, CV_8UC1);
	}
	else confidence = 255*cv::Mat::ones(im[0].size(), CV_8UC1);
}

System::~System()
{
	delete mSphsweep;
}

void System::init()
{
	auto total_time = 0;
	/* SPHORB feature matching */
	cout << "SPHORB feature matching ..." << endl;
	auto tp1 = std::chrono::high_resolution_clock::now();
	SPHORB sorb(10000);

	cv::Mat* descriptors = new cv::Mat[nCam];
	vector<cv::KeyPoint>* kPoints = new vector<cv::KeyPoint>[nCam];

	for(int i = 0; i < nCam; i++)
	{
		sorb(im1K[i], cv::Mat(), kPoints[i], descriptors[i]);
		cout << "Keypoint" << i << ":" << kPoints[i].size() << "\t";
	}
	cout << endl;

	cv::BFMatcher matcher(NORM_HAMMING, false);
	Matches* matches = new Matches[nCam - 1];
	vector<Matches>* dupMatches = new vector<Matches>[nCam - 1];

	for(int i = 0; i < nCam - 1; i++)
	{
		matcher.knnMatch(descriptors[0], descriptors[i + 1], dupMatches[i], 2);
		ratioTest(dupMatches[i], ratio, matches[i]);
	}
	delete[] descriptors;
	delete[] dupMatches;
	descriptors = NULL;
	dupMatches = NULL;

	for(int i = 0; i < nCam - 1; i++)
	{
		cout<<"Matched points: "<<matches[i].size()<<endl;
	}
	if(matches[0].size() < 8)
	{
		cout << "More than 8 matched points are needed!!" << endl;
		return;
	}
	auto tp2 = std::chrono::high_resolution_clock::now();
	auto t_duration = chrono::duration_cast<chrono::milliseconds>( tp2 - tp1 ).count();
	total_time += t_duration;
	cout<<"Duration time: "<<t_duration<<"ms"<<endl<<endl;
/*
	cv::Mat imgMatches;
	::drawMatches(im1K[0], kPoint1, im1K[1], kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,true);
	cv::imwrite("1_matches.jpg", imgMatches);
*/

	/* Coordinate conversion */
	Coordinate** kpCoordPs = new Coordinate*[nCam - 1];
	for(int i = 0; i < nCam - 1; i++)
	{
		kpCoordPs[i] = new Coordinate(im1K[0].rows, im1K[0].cols, kPoints[0], kPoints[i + 1], matches[i]);
		kpCoordPs[i]->ImgToCartesian();
	}

    /* Coordinate conversion */
	/*
    Coordinate *kpCoord1 = new Coordinate(im1K[0].rows, im1K[0].cols, kPoint1, kPoint2, matches1);
	kpCoord1->ImgToCartesian();
	Coordinate *kpCoord2 = new Coordinate(im1K[0].rows, im1K[0].cols, kPoint1, kPoint3, matches2);
	kpCoord2->ImgToCartesian();
        Coordinate *kpCoord3 = new Coordinate(im1K[0].rows, im1K[0].cols, kPoint1, kPoint4, matches3);
	kpCoord3->ImgToCartesian();
	*/

	/* Sfm */
	cout<<"Computing Orientation ..."<<endl;
	tp1 = std::chrono::high_resolution_clock::now();
	//cv::Mat R1, R2, t1, t2;
	{
		for(int i = 0; i < nCam - 1; i++)
		{
			cv::Mat tmp;
			SfM *sfm1 = new SfM(kpCoordPs[i]->xyz1, kpCoordPs[i]->xyz2);
			sfm1->run(maxIter, errTh, perTh);
			sfm1->getRotationMat(tmp); // return target->reference
			R.push_back(tmp);
			sfm1->getTranslationVec(tmp); // return target->reference
			t.push_back(tmp);
			delete sfm1;
		}
		
		//free kpCoordPs
		/*
		for(int i = 0; i < nCam - 1; i++)
		{
			delete[] kpCoordPs[i];
		}
		delete[] kpCoordPs;*/

		/*
		cv::Mat tmp;
		SfM *sfm1 = new SfM(kpCoord1->xyz1, kpCoord1->xyz2);
		sfm1->run(maxIter, errTh, perTh);
		sfm1->getRotationMat(tmp); // return target->reference
		R.push_back(tmp);
		sfm1->getTranslationVec(tmp); // return target->reference
		t.push_back(tmp);
		delete sfm1;
		delete kpCoord1;

		SfM* sfm2 = new SfM(kpCoord2->xyz1, kpCoord2->xyz2);
		sfm2->run(maxIter, errTh, perTh);
		sfm2->getRotationMat(tmp); // return target->reference
		R.push_back(tmp);
		sfm2->getTranslationVec(tmp); // return target->reference
		t.push_back(tmp);
		delete sfm2;
		delete kpCoord2;

                SfM* sfm3 = new SfM(kpCoord3->xyz1, kpCoord3->xyz2);
		sfm3->run(maxIter, errTh, perTh);
		sfm3->getRotationMat(tmp); // return target->reference
		R.push_back(tmp);
		sfm3->getTranslationVec(tmp); // return target->reference
		t.push_back(tmp);
		delete sfm3;
		delete kpCoord3;
		*/

	}                
	tp2 = std::chrono::high_resolution_clock::now();
	t_duration = chrono::duration_cast<chrono::milliseconds>( tp2 - tp1 ).count();
	total_time += t_duration;
    cout<<"Duration time: "<<t_duration<<"ms"<<endl<< endl;
	

	/* Rotate images */
	cout<<"Rotate images ..."<<endl;
	tp1 = std::chrono::high_resolution_clock::now();
/*	
	// for comparision
	double data[3][3] = { {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0} };
	R = cv::Mat(3,3,CV_64FC1, data);
	cv::Mat fix, weighted, closest;
	rotateImg(img2, fix, R, getPixel::FIX);
	rotateImg(img2, weighted, R, DEMO::Pixel::MEAN);
	rotateImg(img2, closest, R, DEMO::Pixel::CLOSEST);
	cv::imwrite("fix.png", fix);
	cv::imwrite("weighted.png", weighted);
	cv::imwrite("closest.png", closest);
*/

	// rotate images
	vector<cv::Mat> imgs;
	imgs.push_back(im[0]);
	for(int i = 1; i < nCam; ++i)
	{	
		decomposeRotationMat(R[i]);
		cout << "t: " << t[i] << endl;
		cv::Mat tmp;
		if (rigid)
		{
			tmp = im[i];
			t[i].at<double>(0) = 0;
			t[i].at<double>(1) = 0;
			t[i].at<double>(2) = -1;
			R[i] = (cv::Mat_<double>(3,3) << 1.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0,  0.0,  1.0);
		}
		else rotateImg(im[i], tmp, R[i], rMethod);
		cout << "Rx: " << R[i] << endl;
		imgs.push_back(tmp);
		cv::imwrite(cv::format("rotated%d.png",i), imgs[i]);
	}


/*
	double rx = 0.175, ry = 0.261799388, rz = 0.175;
	
	cv::Mat tmpx = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx));
	cv::Mat tmpy = (cv::Mat_<double>(3, 3) << cos(ry), 0 , sin(ry), 0 , 1, 0, -sin(ry), 0, cos(ry));
	cv::Mat tmpz = (cv::Mat_<double>(3, 3) << cos(rz), -sin(rz) , 0, sin(rz) , cos(rz), 0, 0, 0, 1);
	cv::Mat _tmp;
	cv::Mat tempr = tmpx * tmpy * tmpz;
	rotateImg(im[0], _tmp, tempr, rMethod);
	cv::imwrite(cv::format("rotated_%f_%f_%f.png",rx, ry, rz), _tmp);
*/

	t = allT(R, t, kPoints, matches);
	
	for(int i = 0; i < nCam; ++i)
	{
		std::cout << i << "th ratio t: " << t[i].t() << std::endl;
	}

	delete[] kPoints;
	delete[] matches;
	kPoints = NULL;	
	matches = NULL;

	// bilateral filter
	for(int i = 0; i < nCam; ++i)
	{
		cv::Mat tmp;
		cv::bilateralFilter(imgs[i], tmp, 3, 25, 25);
		tmp.copyTo(this->im[i]);
		cv::imwrite(cv::format("filtered%d.png",i), this->im[i]);
	}
	tp2 = std::chrono::high_resolution_clock::now();
	t_duration = chrono::duration_cast<chrono::milliseconds>( tp2 - tp1 ).count();
	total_time += t_duration;
	cout << "Duration time: " << t_duration << "ms" << endl << endl;


	// SphereSweeping
	cout << "Sphere Sweeping ..." << endl;
	tp1 = std::chrono::high_resolution_clock::now();

	if(idx != 0)
	{
		swapT();
		swapCam();
	}
	
	mSphsweep->setT(this->t);
	mSphsweep->setFrame(this->im);
	mSphsweep->run(imDepth);
	//cv::imwrite("depthd.png", imDepth);
	//cv::Mat ys_temp;
	//imDepth.copyTo(ys_temp);
	//ys_temp.convertTo(ys_temp, CV_32FC1, 1);
	imDepth.convertTo(imDepth, CV_16UC1, 65536/maxDepth);
	//imDepth.convertTo(imDepth, CV_16UC1, 65535/(10.0 / 0.06));
	//cv::imwrite(cv::format("/home/vcl/Kun/VCL/classroom02/output/cam4/%03d.png", nFrame+1), imDepth);

	// filter depth
	//cv::Mat confidence = 255*cv::Mat::ones(im[0].size(), CV_8UC1);
	double spatialSigma = 8.0;
    double lumaSigma = 8.0;
    double chromaSigma = 8.0;
	cv::ximgproc::fastBilateralSolverFilter(im[idx],imDepth,confidence,imDepth,spatialSigma,lumaSigma,chromaSigma,100);
	//cv::ximgproc::fastBilateralSolverFilter(im[idx],ys_temp,confidence,ys_temp,spatialSigma,lumaSigma,chromaSigma,100);
	//cv::FileStorage fs_w("out.xml", cv::FileStorage::WRITE);
	//fs_w << "ys_temp" << ys_temp;
	//fs_w.release();

	vDepth.push_back(imDepth);
	++nFrame;
	

	tp2 = std::chrono::high_resolution_clock::now();
	t_duration = chrono::duration_cast<chrono::milliseconds>( tp2 - tp1 ).count();
	total_time += t_duration;
	cout<<"Duration time: "<<t_duration<<"ms"<<endl<<endl;

	cout<<"Total time: "<<total_time<<"ms"<<endl;
	cout<<"Frame: "<<nFrame<<endl;
	cv::imwrite("sampledepth.png", imDepth);
}

void System::pushNextSequences(const vector<cv::Mat> &im)
{
	// rotate images
	//im[0].copyTo(this->im[0]);
	cv::bilateralFilter(im[0], this->im[0], 3, 25, 25);
	for(int i = 1; i < nCam; ++i)
	{
		cv::Mat tmp;
		rotateImg(im[i], tmp, R[i], rMethod);
		cv::bilateralFilter(tmp, this->im[i], 3, 25, 25);
		//cv::imwrite(cv::format("rotated%d.png",i), this->im[i]);
	}

	// SphereSweeping
	if(idx != 0) swapCam();
	mSphsweep->nextFrame(this->im);
	mSphsweep->run(imDepth);
	//cv::imwrite("depthd.png", imDepth);
	//imDepth.convertTo(imDepth, CV_8UC1, 255.0/maxDepth);
	imDepth.convertTo(imDepth, CV_16UC1, 65536/maxDepth);
	cv::imwrite(cv::format("/home/vcl/Kun/VCL/classroom02/output/cam4/%03d.png", nFrame+1), imDepth);
	
	// filter depth
	//cv::Mat confidence = 255*cv::Mat::ones(this->im[0].size(), CV_8UC1);
	double spatialSigma = 8.0;
    double lumaSigma = 8.0;
    double chromaSigma = 8.0;
	cv::ximgproc::fastBilateralSolverFilter(this->im[idx],imDepth,confidence,imDepth,spatialSigma,lumaSigma,chromaSigma,100);
	
	vDepth.push_back(imDepth);

	++nFrame;
	
	cout<<"Frame: "<<nFrame<<endl;
}

void System::getDepthSequences(vector<cv::Mat> &vDepth)
{
	vDepth.clear();
	vDepth.resize(nFrame);
	copy(this->vDepth.begin(), this->vDepth.end(), vDepth.begin());
}

void System::swapT()
{
    swap(t[0], t[idx]);
    cv::Mat tmp;
    t[0].copyTo(tmp);
    for(auto i = 0; i < nCam; ++i) t[i] = t[i]-tmp;
}

void System::swapCam()
{
    swap(im[0], im[idx]);
}

void System::setSPHORB_ratio(const float ratio)
{
	this->ratio = ratio;
}
void System::setSfM_maxIter(const int maxIter)
{
	this->maxIter = maxIter;
}
void System::setSfM_errorTh(const double errTh)
{
	this->errTh = errTh;
}
void System::setSfM_percentTh(const double perTh)
{
	this->perTh = perTh;
}
void System::setRotation_method(DEMO::Pixel rMethod)
{
	this->rMethod = rMethod;
}
void System::setSphereSweeping_blockSize(const int blockSize)
{
	this->blockSize = blockSize;
}
void System::setSphereSweeping_minDepth(const double minDepth)
{
	this->minDepth = minDepth;
}
void System::setSphereSweeping_maxDepth(const double maxDepth)
{
	this->maxDepth = maxDepth;
}
void System::setSphereSweeping_interval(const double interval)
{
	this->interval = interval;
	this->numDepth = (int)((maxDepth-minDepth)/this->interval) + 1;
}
void System::setSphereSweeping_numDepth(const int numDepth)
{
	this->numDepth = numDepth;
	this->interval = (maxDepth-minDepth)/(double)this->numDepth;
}
void System::setSphereSweeping_method(const DEMO::Pixel ssMethod)
{
	this->ssMethod = ssMethod;
}
void System::decomposeRotationMat(cv::Mat R1)
{
    double x1 = atan2(R1.at<double>(2,1), R1.at<double>(2,2));
    double y1 = atan2( -1.0 * R1.at<double>(2,0), \
        sqrt(pow(R1.at<double>(2,1), 2.0)+pow(R1.at<double>(2,2), 2.0)));
    double z1 = atan2(R1.at<double>(1,0), R1.at<double>(0,0));

    cout << "realx: " << x1 * (180.0 / M_PI) << 
    "\trealy: " << y1 * (180.0 / M_PI) << \
    "\trealz: " << z1 * (180.0 / M_PI) << endl;
}


vector<int*> System::allMatch(vector<cv::KeyPoint>* kPoints, Matches* matches)
{
	vector<int*> indexing;
	int* temp = new int[nCam - 1];
	int cnt = 1;
	
	for (int j = 0; j < static_cast<int>(matches[0].size()); j ++)
	{	
		for (int i = 0; i < static_cast<int>(matches[cnt - 1].size()); i++)
		{
			//if(static_cast<int>(kPoints[0][matches[0][j].queryIdx].pt.y) == static_cast<int>(kPoints[0][matches[cnt - 1][i].queryIdx].pt.y) && (static_cast<int>(kPoints[0][matches[0][j].queryIdx].pt.x) == static_cast<int>(kPoints[0][matches[cnt - 1][i].queryIdx].pt.x)))
			//if(abs(kPoints[0][matches[0][j].queryIdx].pt.y - kPoints[0][matches[cnt - 1][i].queryIdx].pt.y) < 2 && abs(kPoints[0][matches[0][j].queryIdx].pt.x - kPoints[0][matches[cnt - 1][i].queryIdx].pt.x) < 2)
			if(abs(kPoints[0][matches[0][j].queryIdx].pt.y - kPoints[0][matches[cnt - 1][i].queryIdx].pt.y) < 10 && abs(kPoints[0][matches[0][j].queryIdx].pt.x - kPoints[0][matches[cnt - 1][i].queryIdx].pt.x) < 10)
			{
				temp[cnt - 1] = i;
				cnt = cnt + 1;
				if (cnt == nCam) 
				{
					cnt = 1;
					indexing.push_back(temp);
					temp = new int[nCam - 1];
					break;
				}
				else 
				{
					j = j - 1; 
					break;
				}
			}
			//else if(i == static_cast<int>(matches[cnt - 1].size()) - 1) cnt = 1;
		}
	}
	if (cnt != nCam) delete[] temp;
	std::cout << "\nAll matched points: " << indexing.size() << std::endl << std::endl;
	
	return indexing;
}

cv::Mat System::allRot(std::vector<cv::Mat> R, vector<cv::KeyPoint>* kPoints, Matches* matches, std::vector<int*> all_match)
{
	const int h = 640;
	const int w = 1280;
	cv::Mat match_mat = cv::Mat::zeros(static_cast<int>(all_match.size()), nCam, CV_64FC2);
	cv::Mat cart = cv::Mat::zeros(static_cast<int>(all_match.size()), nCam, CV_64FC3);

	#pragma omp parallel for
	for (int j = 0; j < nCam; j++)
	{
		for (int i = 0; i < static_cast<int>(all_match.size()); i++)
		{
			if (j != 0) 
			{
				match_mat.at<Vec2d>(i, j)[0] = kPoints[j][matches[j - 1][all_match[i][j - 1]].trainIdx].pt.x; 
				match_mat.at<Vec2d>(i, j)[1] = kPoints[j][matches[j - 1][all_match[i][j - 1]].trainIdx].pt.y;
			}
			else 
			{
				match_mat.at<Vec2d>(i, j)[0] = kPoints[j][matches[j][all_match[i][0]].queryIdx].pt.x;
				match_mat.at<Vec2d>(i, j)[1] = kPoints[j][matches[j][all_match[i][0]].queryIdx].pt.y;
			}
		}
	}

	#pragma omp parallel for
	for (int j = 0; j < nCam; j++)
	{
		for (int i = 0; i < static_cast<int>(all_match.size()); i++)
		{
			cart.at<Vec3d>(i, j)[0] = sin(match_mat.at<Vec2d>(i, j)[1] / h * (M_PI)) * cos(match_mat.at<Vec2d>(i, j)[0] / w * (2 * M_PI));
			cart.at<Vec3d>(i, j)[1] = sin(match_mat.at<Vec2d>(i, j)[1] / h * (M_PI)) * sin(match_mat.at<Vec2d>(i, j)[0] / w * (2 * M_PI));
			cart.at<Vec3d>(i, j)[2] = cos(match_mat.at<Vec2d>(i, j)[1] / h * (M_PI));
		}
	}

	#pragma omp parallel for
	for (int j = 1; j < nCam; j++)
	{
		//cv::transpose(R[j], R[j]);
		for (int i = 0; i < static_cast<int>(all_match.size()); i++)
		{
			cv::Mat temp = R[j] * cart.at<Vec3d>(i, j);
			cart.at<Vec3d>(i, j)[0] = temp.at<double>(0, 0);
			cart.at<Vec3d>(i, j)[1] = temp.at<double>(1, 0);
			cart.at<Vec3d>(i, j)[2] = temp.at<double>(2, 0);
		}
	}

	/////////////////////////////////////////////////////////////
	/*
	for (int j = 0; j < nCam; j++)
	{
		for (int i = 0; i < static_cast<int>(all_match.size()); i++)
		{
			match_mat.at<Vec2d>(i, j)[1] = acos(cart.at<Vec3d>(i, j)[2]) * h / (M_PI);
			double temp_phi = atan2(cart.at<Vec3d>(i, j)[1], cart.at<Vec3d>(i, j)[0]);
			if (temp_phi < 0) match_mat.at<Vec2d>(i, j)[0] = ((2 * M_PI) + temp_phi) * w / (2 * M_PI);
			else match_mat.at<Vec2d>(i, j)[0] = (temp_phi) * w / (2 * M_PI);
		}
	}
	*/
	/////////////////////////////////////////////////////////////

	return cart;
}

std::vector<cv::Mat> System::allT(std::vector<cv::Mat> R, std::vector<cv::Mat> t, vector<cv::KeyPoint>* kPoints, Matches* matches)
{
	if (static_cast<int>(t.size()) <= 2) return t;
	std::vector<int*> all_match = allMatch(kPoints, matches);
	cv::Mat cart = allRot(R, kPoints, matches, all_match);
	cv::Mat depth = cv::Mat::zeros(all_match.size(), nCam, CV_64FC1);
	cv::Mat ratio = cv::Mat::ones(all_match.size(), nCam, CV_64FC1) * (-1);

	#pragma omp parallel for
	for (int j = 1; j < depth.cols; j++)
	{
		for (int i = 0; i < depth.rows; i++)
		{
			double th1, th2;
			th1 = acos(cv::Mat(t[j].t() * cart.at<Vec3d>(i, 0)).at<double>(0, 0));
			th2 = acos(cv::Mat((-t[j]).t() * cart.at<Vec3d>(i, j)).at<double>(0, 0));
			depth.at<double>(i, j) = sin(th2) / sin(th1 + th2);
			//std::cout << th1 << "    " << th2 << std::endl;
		}
	}

	#pragma omp parallel for
	for (int j = 2; j < depth.cols; j++)
	{
		for (int i = 0; i < depth.rows; i++)
		{
			if (depth.at<double>(i, 1) > 0 && depth.at<double>(i, j)) // if depth were negative, discard that
			{
				ratio.at<double>(i, j) = depth.at<double>(i, 1) / depth.at<double>(i, j);
			}
		}
	}

	/*
	for (int i = 0; i < depth.rows; i++)
	{
		for (int j = 0; j < depth.cols; j++)
		{
			std::cout << ratio.at<double>(i, j) << "    ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	*/
	/*
	for (int j = 2; j < depth.cols; j++)
	{
		double sum = 0, cnt = 0;
		for (int i = 0; i < depth.rows; i++)
		{
			if (ratio.at<double>(i, j) < 7 && ratio.at<double>(i, j) > 0)
			{cout << "=======================" << endl;
				sum = sum + abs(ratio.at<double>(i, j));
				cnt = cnt + 1;
			}
		}
		t[j] = t[j] * sum / cnt;
		std::cout << "ratio is " << sum / cnt << std::endl;
	}	
	std::cout << std::endl;
	*/


	double ratio_cand[nCam];
	int ratio_num[nCam];

	std::fill_n(ratio_cand, nCam, 1);
	std::fill_n(ratio_num, nCam, 0);
	#pragma omp parallel for
	for(int iter = 0; iter < depth.rows; iter++)
	{
		for (int j = 2; j < depth.cols; j++)
		{
			for (int i = 0; i < depth.rows; i++)
			{
				double cand = ratio.at<double>(i, j);
				if (cand < 0 || cand > maxDepth) {}
				else
				{
					int cnt = 0;
					//for (int k = 0; k < depth.rows; k++) if (ratio.at<double>(k, j) > 0 && std::abs(ratio.at<double>(k, j) - cand) < 0.3) cnt = cnt + 1;
					for (int k = 0; k < depth.rows; k++) if (ratio.at<double>(k, j) > 0 && std::abs(ratio.at<double>(k, j) - cand) < 0.01) cnt = cnt + 1;
					if (cnt > ratio_num[j]) 
					{
						ratio_num[j] = cnt;
						ratio_cand[j] = cand;
					}
				}
			}
		}
	}

	for (int j = 0; j < depth.cols; j++)
	{
		std::cout << ratio_cand[j] << "    ";
		t[j] = t[j] * ratio_cand[j];
	}
	std::cout << std::endl;

	return t;
}

} //namespace DEMO
