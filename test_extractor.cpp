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

		// Columns
		const int n_cols = img.cols;
		const int n_rows = img.rows;
		const int n_bins_u = 20;
		const int n_bins_v = 12;
		const int num_maximum_feature_per_bin = 1;
		const int thres_fast = 15;
		const int radius = 5;

		// Matching parameters
		const size_t grid_size_u_in_pixel = 20;
		const size_t grid_size_v_in_pixel = 20;
		const double search_radius = 30.0;
		const int threshold_descriptor_distance = 50;

		// Generate modules
		std::unique_ptr<FeatureExtractor> feature_extractor = std::make_unique<FeatureExtractor>();
		feature_extractor->initParams(n_cols, n_rows, n_bins_u, n_bins_v, thres_fast, radius);
		
		std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();

		std::vector<cv::KeyPoint> kpts_selected;
		std::vector<cv::Mat> desc_selected;
		feature_extractor->extractAndComputORBwithBinning(
			img, n_bins_u, n_bins_v, num_maximum_feature_per_bin,
			kpts_selected, desc_selected);

		std::vector<cv::KeyPoint> kpts_all;
		std::vector<cv::Mat> desc_all;
		feature_extractor->extractAndComputeORB(img, kpts_all, desc_all);

		// Estimate scale level for features
		// 처음 관측된 local depth 대비 현재 local depth 의 비율을 
		// scale2 = scale1 * (d1 / d2) = scale1 * (d1 / (d1 - t12.z()))
		// scale_factor = scale2/scale1 = d1 / (d1 - t12.z())
		// delta_scale = log2(scale_factor)

		Eigen::Vector3d t12({0,0, 0.3});
		double depth1 = 1.23;
		double depth2 = depth1 - t12.z();
		
		std::vector<int> estimated_scale_level_selected;
		estimated_scale_level_selected.reserve(kpts_selected.size());
		for(size_t index = 0; index < kpts_selected.size(); ++index) {
			const int& scale_level_original = kpts_selected[index].octave;
			const int changed_scale = std::round(log2(depth2/depth1));
			estimated_scale_level_selected.push_back(scale_level_original + changed_scale);
		}

		// Descriptor		
		std::unordered_map<int, int> association_projected_to_reference;
		feature_matcher->matchByDescriptorWithEstimatedScale(
			kpts_selected, desc_selected,	estimated_scale_level_selected,
			kpts_all, desc_all, 
			n_cols, n_rows,
			grid_size_u_in_pixel, grid_size_v_in_pixel, search_radius, 
			threshold_descriptor_distance,
			association_projected_to_reference);

		// Draw matching
		cv::Mat img_hconcat;
		cv::hconcat(img_color, img_color, img_hconcat);
		for(const auto& [index_projected, index_reference] : association_projected_to_reference){			
			const cv::Point2f& pt_selected = kpts_selected[index_projected].pt;
			cv::Point2f& pt_reference = kpts_all[index_reference].pt;
			pt_reference.x += n_cols;

			cv::drawMarker(img_hconcat, pt_selected, cv::Scalar(255,0,0), cv::MARKER_TILTED_CROSS, 10, 1);
			cv::drawMarker(img_hconcat, pt_reference, cv::Scalar(0,255,0), cv::MARKER_SQUARE, 10, 1);
		}

		cv::namedWindow("matching result");
		cv::imshow("matching result", img_hconcat);
		cv::waitKey(0);

		// KLT matching
		const size_t window_size = 25;
		const size_t max_pyramid_level = 4;
		const double threshold_error = 30.0;
		std::vector<cv::Point2f> pts_tracked;
		std::vector<bool> mask_tracked;
		feature_matcher->matchByOpticalFlow(kpts_selected, img, img, 
			window_size, max_pyramid_level, threshold_error,
			pts_tracked, mask_tracked);

		cv::Mat img_hconcat_track;
		cv::hconcat(img_color, img_color, img_hconcat_track);
		for(size_t i = 0; i < kpts_selected.size(); ++i){			
			const cv::Point2f& pt_selected = kpts_selected[i].pt;
			cv::Point2f& pt_reference = pts_tracked[i];
			pt_reference.x += n_cols;

			if( !mask_tracked[i] ) continue;

			cv::line(img_hconcat_track, pt_selected, pt_reference, cv::Scalar(0,255,255),1);
			cv::drawMarker(img_hconcat_track, pt_selected, cv::Scalar(255,0,0), cv::MARKER_SQUARE, 8, 1);
			cv::circle(img_hconcat_track, pt_selected, 1, cv::Scalar(255,0,0), 1);
			cv::drawMarker(img_hconcat_track, pt_reference, cv::Scalar(0,255,0), cv::MARKER_SQUARE, 8, 1);
			cv::circle(img_hconcat_track, pt_reference, 1, cv::Scalar(0,255,0), 1);
		}

		cv::namedWindow("tracking result");
		cv::imshow("tracking result", img_hconcat_track);
		cv::waitKey(0);
	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}