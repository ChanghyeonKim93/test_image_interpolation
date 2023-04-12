#ifndef _KLT_TRACKER_H_
#define _KLT_TRACKER_H_

#include <iostream>

// Eigen
#include "Eigen/Dense"

// OPENCV
#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"

#include "image_processing.h"

#include "aligned_memory.h"

class KLTTracker
{
  using _KLT_numeric = float;
  using Mat11 = Eigen::Matrix<_KLT_numeric, 1, 1>;
  using Mat22 = Eigen::Matrix<_KLT_numeric, 2, 2>;
  using Mat33 = Eigen::Matrix<_KLT_numeric, 3, 3>;
  using Mat44 = Eigen::Matrix<_KLT_numeric, 4, 4>;
  using Mat55 = Eigen::Matrix<_KLT_numeric, 5, 5>;
  using Mat66 = Eigen::Matrix<_KLT_numeric, 6, 6>;
  using Mat77 = Eigen::Matrix<_KLT_numeric, 7, 7>;
  using Mat88 = Eigen::Matrix<_KLT_numeric, 8, 8>;

  using Vec1 = Eigen::Matrix<_KLT_numeric, 1, 1>;
  using Vec2 = Eigen::Matrix<_KLT_numeric, 2, 1>;
  using Vec3 = Eigen::Matrix<_KLT_numeric, 3, 1>;
  using Vec4 = Eigen::Matrix<_KLT_numeric, 4, 1>;
  using Vec5 = Eigen::Matrix<_KLT_numeric, 5, 1>;
  using Vec6 = Eigen::Matrix<_KLT_numeric, 6, 1>;
  using Vec7 = Eigen::Matrix<_KLT_numeric, 7, 1>;
  using Vec8 = Eigen::Matrix<_KLT_numeric, 8, 1>;

public:
  KLTTracker();
  ~KLTTracker();

public:
  void trackForwardAdditive(const cv::Mat &img0, const cv::Mat &img1,
                            const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
                            const size_t max_level, const size_t window_size = 25, const size_t max_iteration = 30,
                            const bool use_sparse_patch = false, const bool use_initial_flow = false);
  // void trackInverseCompositional(const cv::Mat &img0, const cv::Mat &img1,
  //                                const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
  //                                size_t max_level, size_t max_iteration = 30,
  //                                bool is_use_initial_flow = false);

  // void trackHorizontal(const cv::Mat &img0, const cv::Mat &img1,
  //                      const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
  //                      bool is_use_initial_flow = false);

private:
  void allocateAlignedMemory();

  void generatePatchPixels(const size_t window_size,
                           std::vector<cv::Point2f> &patch_pixels, const bool make_sparse_patch = false);

private:
  float *upattern_dense_ = nullptr;
  float *vpattern_dense_ = nullptr;

  float *upattern_sparse_ = nullptr;
  float *vpattern_sparse_ = nullptr;

  float *buf_u_ = nullptr;
  float *buf_v_ = nullptr;
  float *buf_u_warp_ = nullptr;
  float *buf_v_warp_ = nullptr;

  float *buf_I1_ = nullptr;
  float *buf_du1_ = nullptr;
  float *buf_dv1_ = nullptr;

  float *buf_I2_ = nullptr;
  float *buf_du2_ = nullptr;
  float *buf_dv2_ = nullptr;

  float *buf_residual_ = nullptr;
  float *buf_weight_ = nullptr;

  float *err_ssd_ = nullptr;
  float *err_ncc_ = nullptr;
  bool *mask_ = nullptr;

  // Maximum 28 elements// JtWJ = [
  /* 0, *, *, *, *, *;
     1, 6, *, *, *, *;
     2, 7,11, *, *, *;
     3, 8,12,15, *, *;
     4, 9,13,16,18, *;
     5,10,14,17,19,20];
     JtWr = [21,22,23,24,25,26]^t
     err = [27];
  */
  float *SSEData; // [4 * 28]
  float *AVXData; // [8 * 28]

private:
  // update = -JtWJ.inv()*JtWr (update = mJtWJ.ldlt.solve(JtWr))

  // For 2-D translation (u only) (1-dof)
  float JtWJ_11_;
  float mJtWr_11_;

  // For 2-D translation (u,v) (2-dof)
  Mat22 JtWJ_22_;

  // For 2-D Euclidean transform (3-dof)
  Mat22 JtWJ_33_;

  // For 2-D Similarity transform (4-dof)
  Mat44 JtWJ_44_;

  // For 2-D Affine transform (6-dof)
  Mat66 JtWJ_66_;

  // For Homography transform (8-dof)
  Mat88 JtWJ_88_;
};

#endif