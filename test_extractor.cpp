#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

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
		cv::Mat img_color;
		cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);

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

		std::vector<cv::KeyPoint> kpts_selected;
		std::vector<cv::Mat> desc_selected;
		feature_extractor->extractAndComputORBwithBinning(
			img, n_bins_u, n_bins_v, n_maximum_feature_per_bin,
			kpts_selected, desc_selected);

		// Estimate scale level for features
		std::vector<int> estimated_scale_level_selected;

		// Descriptor
		std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();
		
		const size_t grid_size_u = 20;
		const size_t grid_size_v = 20;
		const double finding_radius = 50.0;
		const int threshold_descriptor_distance = 50;
		std::unordered_map<int, int> projected_reference_association;
		feature_matcher->matchByProjection(
			kpts_selected, desc_selected,	estimated_scale_level_selected,
			kpts_all, desc_all, 
			n_cols, n_rows,
			grid_size_u, grid_size_v, finding_radius,
			threshold_descriptor_distance,
			projected_reference_association);

		// Draw matching
		cv::Mat img_hconcat;
		cv::hconcat(img_color, img_color, img_hconcat);
		for(const auto& [index_projected, index_reference] : projected_reference_association){
			std::cout << "matching: " << index_projected << ", " << index_reference << std::endl;
			std::cout << "query/reference level: " << kpts_selected[index_projected].octave <<
				", " << kpts_all[index_reference].octave << std::endl;
			
			const cv::Point2f& pt_selected = kpts_selected[index_projected].pt;
			cv::Point2f& pt_reference = kpts_all[index_reference].pt;
			pt_reference.x += n_cols;

			cv::drawMarker(img_hconcat, pt_selected, cv::Scalar(255,0,0), cv::MARKER_TILTED_CROSS, 10, 1);
			cv::drawMarker(img_hconcat, pt_reference, cv::Scalar(0,255,0), cv::MARKER_SQUARE, 10, 1);
		}

		cv::namedWindow("matching result");
		cv::imshow("matching result", img_hconcat);
		cv::waitKey(0);

	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}