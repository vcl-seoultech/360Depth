#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "System.hpp"

using namespace std;

int main(int argc, char** argv)
{
	if (argc < 7)
    {
        std::cerr << "usage: ./image <the number of cameras> <max depth> <rigid> <save folder path> <0th images folder> <1st images folder> ...\n";
        return -1;
    }

	auto tp1 = std::chrono::high_resolution_clock::now();
    int camera_n = stoi(argv[1]);
	double max_depth = stod(argv[2]);
	double interval = 0.1;
	bool rigid = stoi(argv[3]);
	string save_path = argv[4];
	vector<cv::Mat> im;

	printf("max_depth: %f    interval: %f\n", max_depth, interval);

	vector<cv::String>* fn = new vector<cv::String>[camera_n];
	for(int i = 0; i < camera_n; i ++) glob(argv[i + 6], fn[i], false);
		
	int count = fn[0].size();

	for (int i = 0; i < camera_n; i ++)
		{
			cv::Mat temp = cv::imread(fn[i][0]);
			cv::resize(temp, temp, cv::Size(1024, 512), 0, 0, CV_INTER_NN);
			im.push_back(temp);
		}

	DEMO::System demo0(im, 0, max_depth, interval, rigid);
	demo0.init();

	for (int j = 1; j < count; j ++)
	{
		im.clear();
		//make a sequence
		for (int i = 0; i < camera_n; i ++)
		{
			cv::Mat temp = cv::imread(fn[i][j]);
			cv::resize(temp, temp, cv::Size(1024, 512), 0, 0, CV_INTER_NN);
			im.push_back(temp);
		}
		demo0.pushNextSequences(im);
	}
	
	cout << endl << "write the depthes" << endl;
	vector<int> png_params;
	png_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	png_params.push_back(0);
	vector<cv::Mat> depth;
	demo0.getDepthSequences(depth);
	for (int i = 0; i < (int)depth.size(); i++) 
	{
		cv::imwrite(save_path + "/out" + to_string(i) + ".png", depth.at(i), png_params);
	}

	auto tp2 = std::chrono::high_resolution_clock::now();
	cout << chrono::duration_cast<chrono::milliseconds>( tp2 - tp1 ).count() << "ms" << endl;

	return 0;
}
