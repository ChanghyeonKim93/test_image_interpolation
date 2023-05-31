#ifndef _FEATURE_EXTRACTOR_H_
#define _FEATURE_EXTRACTOR_H_

#include <iostream>
#include <exception>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include <queue>

// ROS eigen
#include "eigen3/Eigen/Dense"

// ROS cv_bridge
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/core/eigen.hpp"
#include "opencv4/opencv2/features2d.hpp"

using Pixel = cv::Point2f;
using PixelVec = std::vector<Pixel>;

class FeatureExtractor;

class FeatureExtractor {
private:
	float threshold_fast_;
	float scale_factor_;
	int num_level_;

	cv::Ptr<cv::ORB> extractor_orb_; // orb extractor

public:
	/// @brief FeatureExtractor class constructor
	FeatureExtractor(const size_t max_num_features = 5000, const float threshold_fast = 15.0, 
		const float scale_factor = 1.2, const int num_level = 8);

	/// @brief FeatureExtractor class destructor
	~FeatureExtractor();

	void extractORB(const cv::Mat& img, PixelVec& pts_extracted);
	void extractORBwithBinning(const cv::Mat& img, 
		const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
		PixelVec& pts_extracted);
	void extractORBfromEmptyBinOnly(const cv::Mat& img,
		const std::vector<cv::Point2f>& pts_exist, 
		const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
		PixelVec& pts_extracted);

	void extractAndComputeORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted);
	void extractAndComputORBwithBinning(
		const cv::Mat& img, const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
		std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted);
	void extractAndComputORBfromEmptyBinOnly(
		const cv::Mat& img,
		const std::vector<cv::Point2f>& pts_exist, 
		const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
		std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted);
};

#endif