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
    const std::vector<cv::KeyPoint>& kpts_projected, 
    const std::vector<cv::Mat>& desc_projected, 
    const std::vector<int>& estimated_scale_level_projected,
    const std::vector<cv::KeyPoint>& kpts_reference, 
    const std::vector<cv::Mat>& desc_reference,
    const size_t num_column, const size_t num_row,
    const size_t grid_size_u, const size_t grid_size_v, const double radius,
    const int threshold_descriptor_distance,
    std::unordered_map<int,int>& projected_reference_association);

  void matchByBagOfWords();

private:
  void generateReferenceIndexGrid(
    const std::vector<cv::KeyPoint>& kpts_reference,
    const size_t num_column, const size_t num_row, 
    const size_t grid_size_u, const size_t grid_size_v,
    std::vector<std::vector<std::vector<int>>>& reference_grid);

  void findCandidateIndexesFromReferenceIndexGrid(const cv::Point2f& pt_query,
    const std::vector<std::vector<std::vector<int>>>& reference_grid,
    const size_t num_column, const size_t num_row, 
    const size_t grid_size_u, const size_t grid_size_v,
    const size_t num_search_cell_u, const size_t num_search_cell_v, const double search_radius,
    std::vector<int>& candidate_indexes);
  
	int descriptorDistance(const cv::Mat& a, const cv::Mat& b);

private:
  std::vector<std::vector<std::vector<int>>> reference_grid_;
};

#endif