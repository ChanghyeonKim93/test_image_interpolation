#ifndef _KLT_TRACKER_H_
#define _KLT_TRACKER_H_

#include <iostream>

// Eigen
#include <eigen3/Eigen/Dense>

// OPENCV
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/video/tracking.hpp"

#include "image_processing.h"

class KLTTracker
{
  using _KLT_numeric = float;
  using Mat11 = _KLT_numeric;
  using Mat22 = Eigen::Matrix<_KLT_numeric,2,2>;
  using Mat33 = Eigen::Matrix<_KLT_numeric,3,3>;
  using Mat44 = Eigen::Matrix<_KLT_numeric,4,4>;
  using Mat55 = Eigen::Matrix<_KLT_numeric,5,5>;
  using Mat66 = Eigen::Matrix<_KLT_numeric,6,6>;

public:
  KLTTracker();

public:
  void setMaxIteration(const size_t max_iter);
  void setUseSparsePatch();
  void setUseDensePatch();

public:
  void track(const cv::Mat &img0, const cv::Mat &img1,
             const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
             bool is_use_initial_flow = false);

  void trackHorizontal(const cv::Mat &img0, const cv::Mat &img1,
                       const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
                       bool is_use_initial_flow = false);
private:
  void allocateAlignedMemory();

private:
  float *buf_pattern_dense_u_;
  float *buf_pattern_dense_v_;

  float *buf_pattern_sparse_u_;
  float *buf_pattern_sparse_v_;

  float *buf_u_;
  float *buf_v_;
  float *buf_warp_u_;
  float *buf_warp_v_;
  float *buf_I2_;
  float *buf_I1_;

private:
  // update = -JtWJ.inv()*JtWr (update = mJtWJ.ldlt.solve(JtWr))

  // For 2-D translation (u only)
  double mJtWJ_11_;
  double JtWr_11_;

  // For 2-D translation (u,v)
  Mat22 mJtWJ_22_;

  // For 2-D Euclidean transform
  Mat22 hessian_33_;

  // For 2-D Similarity transform
  Mat22 hessian_33_;

  // For 2-D Affine transform

  // For Homography transform
};


#endif