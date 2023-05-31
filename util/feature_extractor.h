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

struct ParamsORB {
	int   MaxFeatures{5000}; // % MaxFeatures (300) % The maximum number of features to retain.
	float ScaleFactor{1.1}; // % ScaleFactor (1.2) % Pyramid decimation ratio.
	int   NLevels{8};        // % NLevels (8)% The number of pyramid levels.
	float EdgeThreshold{31}; // % EdgeThreshold (31)% This is size of the border where the features are not detected.
	int   FirstLevel{0};     // % FirstLevel (0)% The level of pyramid to put source image to.
	int   WTA_K{2};          // % WTA_K (2)% The number of points that produce each element of the oriented BRIEF descriptor. (2, 3, 4)
	std::string ScoreType{"Harris"};// % ScoreType (Harris) % Algorithm used to rank features. (Harris, FAST)
	int   PatchSize{31};     // % PatchSize (31) % Size of the patch used by the oriented BRIEF descriptor.
	float FastThreshold{15}; // % FastThreshold (20)% Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
	float r{5};              // % Radius of non maximum suppresion (5)
	int   n_bins_u{-1};
	int   n_bins_v{-1};
};

struct WeightBin {
	std::vector<int> weight ;
  std::vector<std::priority_queue<std::pair<int, float>>> index_per_bin;
	std::vector<int> u_bound;
	std::vector<int> v_bound;
	int n_bins_u;
	int n_bins_v;
	int u_step  ;
	int v_step  ;
	int n_bins_total;

	float inv_u_step_;
	float inv_v_step_;

	WeightBin() {
		u_bound.resize(0);
		v_bound.resize(0);
		n_bins_u = 0;
		n_bins_v = 0;
		u_step = 0;
		v_step = 0;
		inv_u_step_ = 0;
		inv_v_step_ = 0;
		n_bins_total = 0;
	};
	~WeightBin() {};

	void init(int n_cols, int n_rows, int n_bins_u, int n_bins_v) {
		this->n_bins_u = n_bins_u;
		this->n_bins_v = n_bins_v;
		n_bins_total = n_bins_u*n_bins_v;

		weight.resize(this->n_bins_u*this->n_bins_v);
		for (int i = 0; i < this->n_bins_u*this->n_bins_v; ++i) 
			weight[i] = 1;

		this->u_bound.resize(this->n_bins_u + 1);
		this->v_bound.resize(this->n_bins_v + 1);

		this->u_bound[0] = 1;
		this->v_bound[0] = 1;

		this->u_bound[n_bins_u] = n_cols;
		this->v_bound[n_bins_v] = n_rows;

		this->u_step = (int)floor((float)n_cols / (float)n_bins_u);
		this->v_step = (int)floor((float)n_rows / (float)n_bins_v);

		this->inv_u_step_ = 1.0f/(float)this->u_step;
		this->inv_v_step_ = 1.0f/(float)this->v_step;

		for (int i = 1; i < n_bins_u; ++i) this->u_bound[i] = u_step * i;
		for (int i = 1; i < n_bins_v; ++i) this->v_bound[i] = v_step * i;

		std::cout<< "   - FEATURE_EXTRACTOR - WeightBin - 'init'\n";
	};

	void reset() {
		for (int i = 0; i < n_bins_total; ++i) weight[i] = 1;
	};

	void update(const PixelVec& pts) {
		int n_pts = pts.size();

		for (const auto& p : pts) {
			int u_idx = floor((float)p.x * this->inv_u_step_);
			int v_idx = floor((float)p.y * this->inv_v_step_);
			int bin_idx = v_idx * n_bins_u + u_idx;

			if(bin_idx >= 0 && bin_idx < n_bins_total) 
				weight[bin_idx] = 0;
		}
		// std::cout <<" - FEATURE_EXTRACTOR - WeightBin - 'update' : # input points: "<<n_pts << "\n";
	};

	/// @brief Find bucket index.
	/// @param pt opencv cv::Point2f
	/// @param u bin u
	/// @param v bin v
	/// @return true: in-image & empty bucket, false: out of image or not required bucket.
	bool findBucket(const Pixel& pt, unsigned long& u, unsigned long& v);
};

struct PointFeature {
  cv::Point2f pt;
  int octave;
  float response;
  float angle;
  unsigned int descriptor[8];

  void setFromCvKeypoint(const cv::KeyPoint& kpt, const cv::Mat& desc_in_cv) {
    pt = kpt.pt;
    octave = kpt.octave;
    response = kpt.response;
    angle = kpt.angle;
    // desc_in_cv
  }  
};

class FeatureExtractor 
{
private:
  // 현재까지 가장 높은 스코어를 저장. 
	struct IndexBin	{
		std::vector<int> index_;
		int   index_max_;
		float max_score_;

		IndexBin() {
			index_.reserve(100);
			index_max_ = -1;
			max_score_ = -1;
		};
	};
	
private:
	std::shared_ptr<WeightBin> weight_bin_;

	ParamsORB  params_orb_;
	cv::Ptr<cv::ORB> extractor_orb_; // orb extractor
	// cv::Ptr<cv::Feature2D> extractor_;

private:
	bool  flag_nonmax_;
	bool  flag_debug_;
	int   n_bins_u_;
	int   n_bins_v_;
	int   THRES_FAST_;
	int   NUM_NONMAX_;
	float r_;

	std::vector<IndexBin> index_bins_;

public:
	/// @brief FeatureExtractor class constructor
	FeatureExtractor();

	/// @brief FeatureExtractor class destructor
	~FeatureExtractor();

	void initParams(int n_cols, int n_rows, int n_bins_u, int n_bins_v, int THRES_FAST, int radius);
	void updateWeightBin(const PixelVec& pts);
	void resetWeightBin();
	void suppressCenterBins();
	void extractORBwithBinning(const cv::Mat& img, PixelVec& pts_extracted, bool flag_nonmax);
	void extractORBwithBinning_fast(const cv::Mat& img, PixelVec& pts_extracted, bool flag_nonmax);

	void extractAndComputeORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted);
	void extractAndComputORBwithBinning(
		const cv::Mat& img, const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
		std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted);

public:
	void setNonmaxSuppression(bool flag_on);

};

#endif