#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "util/timer.h"

#include "util/image_processing.h"
#include "util/feature_extractor.h"

#include "util/image_float.h"

int main()
{
	try{
		// cv::Mat img = cv::imread("C:\\Users\\rlack\\Source\\Repos\\test_img_interp\\test_img_interp\\test_image_interpolation\\Lenna.png", cv::IMREAD_GRAYSCALE);
 		cv::Mat img = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);
		

		int n_cols = img.cols;
		int n_rows = img.rows;
		int n_bins_u = 20;
		int n_bins_v = 12;
		int thres_fast = 15;
		int radius = 5;

		std::unique_ptr<FeatureExtractor> feature_extractor = std::make_unique<FeatureExtractor>();
		feature_extractor->initParams(n_cols, n_rows, n_bins_u, n_bins_v, thres_fast, radius);

		std::vector<cv::KeyPoint> kpts_extracted;
		std::vector<cv::Mat> desc_extracted;
		feature_extractor->extractAndComputeORB(img, kpts_extracted, desc_extracted);

		cv::Mat img_draw;
		cv::drawKeypoints(img, kpts_extracted, img_draw, cv::Scalar(255,0,0));

		cv::namedWindow("image raw");
		cv::imshow("image raw", img);
		cv::waitKey(0);

		cv::namedWindow("image ext");
		cv::imshow("image ext", img_draw);
		cv::waitKey(0);

		// Descriptor
	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}