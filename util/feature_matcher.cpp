#include "feature_matcher.h"

FeatureMatcher::FeatureMatcher() {}

FeatureMatcher::~FeatureMatcher() {}

// void FeatureMatcher::matchByProjection(
//   const std::vector<cv::Point2f>& pts_projected, 
//   const std::vector<cv::Mat>& desc_projected, 
//   const std::vector<cv::Point2f>& pts_reference,
//   const std::vector<cv::Mat>& desc_reference,
//   std::unordered_map<int,int>& projected_reference_association)
// {
//   if(pts_projected.size() == 0 || desc_projected.size() == 0)
//     throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_projected.size() == 0 || desc_projected.size() == 0");
//   if(pts_reference.size() == 0 || desc_reference.size() == 0)
//     throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_reference.size() == 0 || desc_reference.size() == 0");
//   if(pts_projected.size() != desc_projected.size())
//     throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_projected.size() != desc_projected.size()");
//   if(pts_reference.size() != desc_reference.size())
//     throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_reference.size() != desc_reference.size()");
  
//   const size_t n_pts_projected = pts_projected.size();
//   const size_t n_pts_reference = pts_reference.size();

//   std::cout << "n_pts_projected : " << n_pts_projected << std::endl;
//   std::cout << "n_pts_reference : " << n_pts_reference << std::endl;
// }

void FeatureMatcher::matchByProjection(
  const std::vector<cv::KeyPoint>& kpts_projected,
  const std::vector<cv::Mat>& desc_projected,
  const std::vector<cv::KeyPoint>& kpts_reference,
  const std::vector<cv::Mat>& desc_reference,
  const size_t grid_size_u, const size_t grid_size_v, const double radius,
  std::unordered_map<int,int>& projected_reference_association)
{
  if(kpts_projected.size() == 0 || desc_projected.size() == 0)
    throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_projected.size() == 0 || desc_projected.size() == 0");
  if(kpts_reference.size() == 0 || desc_reference.size() == 0)
    throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_reference.size() == 0 || desc_reference.size() == 0");
  if(kpts_projected.size() != desc_projected.size())
    throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_projected.size() != desc_projected.size()");
  if(kpts_reference.size() != desc_reference.size())
    throw std::runtime_error("In FeatureMatcher::matchByProjection, pts_reference.size() != desc_reference.size()");

  const size_t n_pts_projected = kpts_projected.size();
  const size_t n_pts_reference = kpts_reference.size();

  std::cout << "n_pts_projected : " << n_pts_projected << std::endl;
  std::cout << "n_pts_reference : " << n_pts_reference << std::endl;

  // 
  this->generateReferenceIndexGrid(kpts_reference, grid_size_u, grid_size_v);

  
}

void FeatureMatcher::generateReferenceIndexGrid(
  const std::vector<cv::KeyPoint>& kpts_reference,
  const size_t grid_size_u, const size_t grid_size_v) {
    
  const size_t n_pts = kpts_reference.size();
  std::cout << "n_pts: " << n_pts << std::endl;


  // reference_grid_.resize()
  // TODO(@): 

}