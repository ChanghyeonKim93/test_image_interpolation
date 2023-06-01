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
	try {
		// cv::Mat img = cv::imread("C:\\Users\\rlack\\Source\\Repos\\test_img_interp\\test_img_interp\\test_image_interpolation\\Lenna.png", cv::IMREAD_GRAYSCALE);
 		cv::Mat img = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);
		
		cv::Mat img1, img2;
		double shift = 0;
		double scale_true = 1.2;

		int image_size = 400;
		int image_half_size = image_size/2;
		int image_center = image_size/2;
		const cv::Point2f tl_original = cv::Point2f(0, 0);
		const cv::Point2f br_original = cv::Point2f(image_size, image_size);
		
		std::cout << "image_center:"<<image_center<<std::endl;
		std::cout << "(double)image_size/scale_true: " << (double)image_half_size/scale_true << std::endl;
		const cv::Point2f tl_changed = cv::Point2f(image_center - (double)image_half_size/scale_true + shift, image_center - (double)image_half_size/scale_true + shift);
		const cv::Point2f br_changed = cv::Point2f(image_center + (double)image_half_size/scale_true + shift, image_center + (double)image_half_size/scale_true + shift);

		std::cout<< tl_changed << std::endl;
		std::cout<< br_changed << std::endl;

		img1 = img(cv::Rect(tl_original, br_original));
		img2 = img(cv::Rect(tl_changed, br_changed));

		cv::resize(img2,img2,cv::Size(image_size,image_size));

		cv::Mat img_tmp = img1;
		img1 = img2;
		img2 = img_tmp;

		cv::namedWindow("img1");
		cv::imshow("img1", img1);
		cv::namedWindow("img2");
		cv::imshow("img2", img2);
		cv::waitKey(0);

		cv::Mat img1_color, img2_color;
		cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
		cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

		// Columns
		const int n_cols = img1.cols;
		const int n_rows = img1.rows;
		const int n_bins_u = 20;
		const int n_bins_v = 12;
		const int num_maximum_feature_per_bin = 1;

		// Matching parameters
		const size_t grid_size_u_in_pixel = 20;
		const size_t grid_size_v_in_pixel = 20;
		const double search_radius = 10.0;
		const int threshold_descriptor_distance = 50;

		// KLT parameters
		const size_t window_size = 15;
		const size_t max_pyramid_level = 5;
		const double threshold_error = 15.0;

		// Generate modules
		const size_t max_num_features = 10000;
		const float threshold_fast = 30.0;
		const float scale_factor = 1.04;
		const int num_level = 12;
		std::unique_ptr<FeatureExtractor> feature_extractor = 
			std::make_unique<FeatureExtractor>(max_num_features, threshold_fast, scale_factor, num_level);
		
		std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();

		std::vector<cv::KeyPoint> kpts_selected;
		std::vector<cv::Point2f> pts_track_true;
		std::vector<cv::Mat> desc_selected;
		feature_extractor->extractAndComputORBwithBinning(
			img1, n_bins_u, n_bins_v, num_maximum_feature_per_bin,
			kpts_selected, desc_selected);

		for(size_t i = 0; i < kpts_selected.size(); ++i){
			pts_track_true.push_back( kpts_selected[i].pt);
			pts_track_true[i].x = (pts_track_true[i].x - image_center)/scale_true + image_center - shift;
			pts_track_true[i].y = (pts_track_true[i].y - image_center)/scale_true + image_center - shift;

		}

		std::vector<cv::KeyPoint> kpts_all;
		std::vector<cv::Mat> desc_all;
		feature_extractor->setThresholdFast(5.0);
		feature_extractor->extractAndComputeORB(img2, kpts_all, desc_all);

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
			int estimated_scale = scale_level_original - changed_scale;
			if(estimated_scale < 0) estimated_scale = 0;
			if(estimated_scale > 12) estimated_scale = 12;
			estimated_scale_level_selected.push_back(estimated_scale);
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
		cv::hconcat(img1_color, img2_color, img_hconcat);
		for(const auto& [index_projected, index_reference] : association_projected_to_reference){			
			const cv::Point2f& pt_selected = kpts_selected[index_projected].pt;
			cv::Point2f pt_reference = kpts_all[index_reference].pt;
			pt_reference.x += n_cols;

			cv::drawMarker(img_hconcat, pt_selected, cv::Scalar(255,0,0), cv::MARKER_TILTED_CROSS, 10, 1);
			cv::drawMarker(img_hconcat, pt_reference, cv::Scalar(0,255,0), cv::MARKER_SQUARE, 10, 1);
		}

		cv::namedWindow("matching result");
		cv::imshow("matching result", img_hconcat);
		cv::waitKey(0);

		// KLT matching
		std::vector<cv::Point2f> pts_tracked;
		std::vector<bool> mask_tracked;
		feature_matcher->matchByOpticalFlow(kpts_selected, img1, img2, 
			window_size, max_pyramid_level, threshold_error,
			pts_tracked, mask_tracked);

		cv::Mat img_hconcat_track;
		cv::hconcat(img1_color, img2_color, img_hconcat_track);
		for(size_t i = 0; i < kpts_selected.size(); ++i){			
			const cv::Point2f& pt_selected = kpts_selected[i].pt;
			cv::Point2f pt_reference = pts_tracked[i];
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


		// Descriptor	matching with klt prior
		std::unordered_map<int, int> association_projected_to_reference2;
		std::vector<cv::KeyPoint> kpts_tracked;
		for(size_t i = 0; i < pts_tracked.size(); ++i){
			cv::KeyPoint kpt = kpts_selected[i];
			kpt.pt = pts_tracked[i];
			kpts_tracked.push_back(kpt);
		}

		feature_matcher->matchByDescriptorWithEstimatedScale(
			kpts_tracked, desc_selected, estimated_scale_level_selected,
			kpts_all, desc_all, 
			n_cols, n_rows,
			grid_size_u_in_pixel, grid_size_v_in_pixel, search_radius, 
			threshold_descriptor_distance,
			association_projected_to_reference2);

		std::cout << "association_projected_to_reference2.size() : " << association_projected_to_reference2.size() << std::endl;

		cv::Mat img_hconcat_track2;
		cv::hconcat(img1_color, img2_color, img_hconcat_track2);
		for(const auto& [index_projected, index_reference] : association_projected_to_reference2){			
			const cv::Point2f& pt_selected = kpts_selected[index_projected].pt;
			cv::Point2f pt_matched = kpts_all[index_reference].pt;
			pt_matched.x += n_cols;

			cv::Point2f pt_tracked = pts_tracked[index_projected];
			pt_tracked.x += n_cols;

			cv::Point2f pt_true = pts_track_true[index_projected];
			pt_true.x += n_cols;

			// cv::line(img_hconcat_track2, pt_matched,pt_selected, cv::Scalar(0,255,255),1);
			cv::drawMarker(img_hconcat_track2, pt_selected, cv::Scalar(255,0,0), cv::MARKER_TILTED_CROSS, 10, 1);
			cv::circle(img_hconcat_track2, pt_selected, 1, cv::Scalar(255,0,0), 1);


			cv::drawMarker(img_hconcat_track2, pt_tracked, cv::Scalar(0,255,0), cv::MARKER_SQUARE, 7, 1);
			cv::circle(img_hconcat_track2, pt_tracked, 1, cv::Scalar(0,255,0), 1);

			cv::drawMarker(img_hconcat_track2, pt_matched, cv::Scalar(0,0,255), cv::MARKER_SQUARE, 7, 1);
			cv::circle(img_hconcat_track2, pt_matched, 1, cv::Scalar(0,0,255), 1);

			cv::drawMarker(img_hconcat_track2, pt_true, cv::Scalar(255,255,0), cv::MARKER_SQUARE, 12, 1);
			cv::circle(img_hconcat_track2, pt_true, 1, cv::Scalar(255,255,0), 1);

			cv::Point2f dp_klt = pt_true - pt_tracked;
			cv::Point2f dp_match = pt_true - pt_matched;
			std::cout << "error klt , match : " << dp_klt.dot(dp_klt) << ", " << dp_match.dot(dp_match) << std::endl;
		}

		cv::namedWindow("tracking result2");
		cv::imshow("tracking result2", img_hconcat_track2);
		cv::waitKey(0);

		// Calculate scale
		for(const auto& [index_projected, index_reference] : association_projected_to_reference2){			
			const double scale_selected = kpts_selected[index_projected].octave;
			const double scale_matched = kpts_all[index_reference].octave;
			std::cout << "scale (selected, matched): " << scale_selected << ", " << scale_matched << std::endl;
		}

	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}