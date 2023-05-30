#ifndef _FEATURE_MATCHER_H_
#define _FEATURE_MATCHER_H_
#include <iostream>

#include <vector>
#include <unordered_map>

#include <opencv4/opencv2/core.hpp>

class FeatureMatcher {
private:

public:
  FeatureMatcher();
  ~FeatureMatcher();

  void matchByProjection(
    const std::vector<cv::Point2f>& pts_projected, 
    const std::vector<cv::Mat>& desc_projected, 
    const std::vector<cv::Point2f>& pts_reference,
    const std::vector<cv::Mat>& desc_reference,
    std::unordered_map<int,int>& projected_reference_association);

  void matchByProjection(
    const std::vector<cv::KeyPoint>& kpts_projected, 
    const std::vector<cv::Mat>& desc_projected, 
    const std::vector<cv::KeyPoint>& kpts_reference,
    const std::vector<cv::Mat>& desc_reference,
    const size_t grid_size_u, const size_t grid_size_v, const double radius,
    std::unordered_map<int,int>& projected_reference_association);

  void matchByBagOfWords();

public:

private:
  void generateReferenceIndexGrid(
    const std::vector<cv::KeyPoint>& kpts_reference,
    const size_t grid_size_u, const size_t grid_size_v
  );

private:
  std::vector<std::vector<std::vector<int>>> reference_grid_;
};

#endif