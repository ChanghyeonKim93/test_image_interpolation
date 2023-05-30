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

#include "util/image_float.h"

#include "util/feature_extractor.h"
#include "util/feature_matcher.h"

#include "util/converter.h"

int main()
{
	try{
		// cv::Mat img = cv::imread("C:\\Users\\rlack\\Source\\Repos\\test_img_interp\\test_img_interp\\test_image_interpolation\\Lenna.png", cv::IMREAD_GRAYSCALE);
 		cv::Mat img = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);
		

		const int n_cols = img.cols;
		const int n_rows = img.rows;
		const int n_bins_u = 20;
		const int n_bins_v = 12;
		const int n_maximum_feature_per_bin = 5;
		const int thres_fast = 15;
		const int radius = 5;

		std::unique_ptr<FeatureExtractor> feature_extractor = std::make_unique<FeatureExtractor>();
		feature_extractor->initParams(n_cols, n_rows, n_bins_u, n_bins_v, thres_fast, radius);

		std::vector<cv::KeyPoint> kpts_all;
		std::vector<cv::Mat> desc_all;
		feature_extractor->extractAndComputeORB(img, kpts_all, desc_all);

		cv::namedWindow("image raw");
		cv::imshow("image raw", img);
		cv::waitKey(0);

		std::vector<cv::KeyPoint> kpts_selected;
		std::vector<cv::Mat> desc_selected;
		feature_extractor->extractAndComputORBwithBinning(
			img, n_bins_u, n_bins_v, n_maximum_feature_per_bin,
			kpts_selected, desc_selected);

		// Draw all features
		cv::Mat img_draw;
		cv::drawKeypoints(img, kpts_all, img_draw, cv::Scalar(255,0,0));
		for(size_t i = 0; i < kpts_selected.size(); ++i){
			const cv::Point2f &pt = kpts_selected[i].pt;
			cv::drawMarker(img_draw, pt, cv::Scalar(0, 200, 0), cv::MARKER_CROSS, 8, 1);
		}

		cv::namedWindow("image ext");
		cv::imshow("image ext", img_draw);
		cv::waitKey(0);

		// Descriptor
		std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();
		
		const size_t grid_size_u = 20;
		const size_t grid_size_v = 20;
		const double finding_radius = 30.0;

		std::unordered_map<int, int> projected_reference_association;
		feature_matcher->matchByProjection(
			kpts_selected, desc_selected,	kpts_all, desc_all, 
			grid_size_u, grid_size_v, finding_radius,
			projected_reference_association);

	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}