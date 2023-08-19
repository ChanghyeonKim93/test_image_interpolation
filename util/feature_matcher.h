#ifndef _FEATURE_MATCHER_H_
#define _FEATURE_MATCHER_H_
#include <iostream>

#include <vector>
#include <unordered_map>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/video.hpp>  // KLT tracker

class FeatureMatcher {
  public:
  FeatureMatcher();
  ~FeatureMatcher();

  void matchByOpticalFlow(
    /* inputs */
    const std::vector<cv::KeyPoint> &kpts1,
    const cv::Mat &image1,
    const cv::Mat &image2,
    /* parameters */
    const size_t window_size,
    const size_t max_pyramid_level,
    const double threshold_error,
    /* outputs */
    std::vector<cv::Point2f> &pts_tracked,
    std::vector<bool> &mask_tracked,
    /* optional */
    bool use_pts_tracked_prior = false);

  void matchByOpticalFlowBidirection(
    /* inputs */
    const std::vector<cv::KeyPoint> &kpts1,
    const cv::Mat &image1,
    const cv::Mat &image2,
    /* parameters */
    const size_t window_size,
    const size_t max_pyramid_level,
    const double threshold_error,
    const double threshold_bidirection_pixel_error,
    /* outputs */
    std::vector<cv::Point2f> &pts_tracked,
    std::vector<bool> &mask_tracked,
    /* optional */
    bool use_pts_tracked_prior = false);

  void matchByDescriptor(
    /* inputs */
    const std::vector<cv::KeyPoint> &kpts_projected,
    const std::vector<cv::Mat> &desc_projected,
    const std::vector<cv::KeyPoint> &kpts_reference,
    const std::vector<cv::Mat> &desc_reference,
    /* parameters */
    const size_t num_column,
    const size_t num_row,
    const size_t grid_size_u_in_pixel,
    const size_t grid_size_v_in_pixel,
    const double radius,
    const int threshold_descriptor_distance,
    /* outputs */
    std::unordered_map<int, int> &projected_reference_association);

  void matchByDescriptorWithEstimatedScale(
    /* inputs */
    const std::vector<cv::KeyPoint> &kpts_projected,
    const std::vector<cv::Mat> &desc_projected,
    const std::vector<int> &estimated_scale_level_projected,
    const std::vector<cv::KeyPoint> &kpts_reference,
    const std::vector<cv::Mat> &desc_reference,
    /* parameters */
    const size_t num_column,
    const size_t num_row,
    const size_t grid_size_u_in_pixel,
    const size_t grid_size_v_in_pixel,
    const double radius,
    const int threshold_descriptor_distance,
    /* outputs */
    std::unordered_map<int, int> &projected_reference_association);

  void matchByBagOfWords();

  private:
  void generateReferenceIndexGrid(
    const std::vector<cv::KeyPoint> &kpts_reference,
    const size_t num_column,
    const size_t num_row,
    const size_t grid_size_u_in_pixel,
    const size_t grid_size_v_in_pixel,
    std::vector<std::vector<std::vector<int>>> &reference_grid);

  void findCandidateIndexesFromReferenceIndexGrid(
    const cv::Point2f &pt_query,
    const std::vector<std::vector<std::vector<int>>> &reference_grid,
    const size_t num_column,
    const size_t num_row,
    const size_t grid_size_u_in_pixel,
    const size_t grid_size_v_in_pixel,
    const size_t num_search_cell_u,
    const size_t num_search_cell_v,
    std::vector<int> &candidate_indexes);

  inline int descriptorDistance(const cv::Mat &a, const cv::Mat &b);

  private:
  std::vector<std::vector<std::vector<int>>> reference_grid_;
};

#endif