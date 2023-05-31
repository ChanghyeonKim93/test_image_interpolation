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
  const std::vector<int>& estimated_scale_level_projected,
  const std::vector<cv::KeyPoint>& kpts_reference,
  const std::vector<cv::Mat>& desc_reference,
  const size_t n_cols, const size_t n_rows, 
  const size_t grid_size_u, const size_t grid_size_v, const double radius,
  const int threshold_descriptor_distance,
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

  if(kpts_projected.size() != estimated_scale_level_projected.size())
    throw std::runtime_error("In FeatureMatcher::matchByProjection, kpts_projected.size() != estimated_scale_level_projected.size()");

  const size_t n_pts_projected = kpts_projected.size();
  const size_t n_pts_reference = kpts_reference.size();

  //
  this->generateReferenceIndexGrid(
    kpts_reference, n_cols, n_rows, grid_size_u, grid_size_v, 
    this->reference_grid_);

  // Matching
  const int num_search_cell_u = 3;
  const int num_search_cell_v = 3;
  for(size_t index = 0; index < n_pts_projected; ++index) {
    const cv::KeyPoint& kpt_query = kpts_projected[index];
    const cv::Mat& descriptor_query = desc_projected[index];

    std::vector<int> candidate_indexes;
    this->findCandidateIndexesFromReferenceIndexGrid(
      kpt_query.pt, reference_grid_,
      n_cols, n_rows, grid_size_u, grid_size_v,
      num_search_cell_u,num_search_cell_v, radius, candidate_indexes);
    
    // find most probable match
    struct IndexDistance {
      int index_reference{-1};
      int distance{255};
    };

    IndexDistance index_dist_first({-1, 255});
    if(candidate_indexes.size() > 0) {
      for(const int index_reference : candidate_indexes){
        const cv::KeyPoint& kpt_reference = kpts_reference[index_reference];
        const cv::Mat& descriptor_reference = desc_reference[index_reference];
        const int descriptor_distance = this->descriptorDistance(descriptor_query, descriptor_reference);

        if(descriptor_distance >= threshold_descriptor_distance) continue; // reject over the threshold

        if(descriptor_distance < index_dist_first.distance){ 
          index_dist_first.index_reference = index_reference;
          index_dist_first.distance = descriptor_distance;
        }
      }
      
      if(index_dist_first.index_reference > -1){
        projected_reference_association[index] = index_dist_first.index_reference;
      }
    }    
  }  
}

void FeatureMatcher::generateReferenceIndexGrid(
  const std::vector<cv::KeyPoint>& kpts_reference,
  const size_t num_column, const size_t num_row, 
  const size_t grid_size_u, const size_t grid_size_v,
  std::vector<std::vector<std::vector<int>>>& reference_grid) 
{
  const double inverse_grid_size_u = 1.0/ static_cast<double>(grid_size_u);
  const double inverse_grid_size_v = 1.0/ static_cast<double>(grid_size_v);
  const size_t num_cell_u = std::floor(static_cast<double>(num_column) * inverse_grid_size_u);
  const size_t num_cell_v = std::floor(static_cast<double>(num_row) * inverse_grid_size_v);

  std::cout << num_cell_u << ", " << num_cell_v << std::endl;
  reference_grid.resize(num_cell_v, std::vector<std::vector<int>>(num_cell_u, std::vector<int>(0)));

  const size_t n_pts = kpts_reference.size();
  for(size_t index = 0; index < n_pts; ++index) {
    const cv::Point2f& pt = kpts_reference[index].pt;
    const int index_cell_u = std::floor(pt.x * inverse_grid_size_u);
    const int index_cell_v = std::floor(pt.y * inverse_grid_size_v);

    if(index_cell_u < 0 || index_cell_u >= num_cell_u) continue;
    if(index_cell_v < 0 || index_cell_v >= num_cell_v) continue;

    reference_grid[index_cell_v][index_cell_u].push_back(index);    
  }
}

void FeatureMatcher::findCandidateIndexesFromReferenceIndexGrid(
  const cv::Point2f& pt_query,
  const std::vector<std::vector<std::vector<int>>>& reference_grid,
  const size_t num_column, const size_t num_row, 
  const size_t grid_size_u, const size_t grid_size_v,
  const size_t num_search_cell_u, const size_t num_search_cell_v, const double search_radius,
  std::vector<int>& candidate_indexes)
{
  const double inverse_grid_size_u = 1.0/ static_cast<double>(grid_size_u);
  const double inverse_grid_size_v = 1.0/ static_cast<double>(grid_size_v);
  const size_t num_cell_u = std::floor(static_cast<double>(num_column) * inverse_grid_size_u);
  const size_t num_cell_v = std::floor(static_cast<double>(num_row) * inverse_grid_size_v);

  const int index_cell_u_center = std::floor(pt_query.x * inverse_grid_size_u);
  const int index_cell_v_center = std::floor(pt_query.y * inverse_grid_size_v);

  if(index_cell_u_center < 0 || index_cell_u_center >= num_cell_u) return;
  if(index_cell_v_center < 0 || index_cell_v_center >= num_cell_v) return;

  const int half_num_search_cell_u = (num_search_cell_u / 2);
  const int half_num_search_cell_v = (num_search_cell_v / 2);

  int search_range_u_min = index_cell_u_center - half_num_search_cell_u;
  int search_range_u_max = index_cell_u_center + half_num_search_cell_u;
  int search_range_v_min = index_cell_v_center - half_num_search_cell_v;
  int search_range_v_max = index_cell_v_center + half_num_search_cell_v;

  if(search_range_u_min < 0) search_range_u_min = 0;
  if(search_range_u_max >= num_cell_u) search_range_u_max = num_cell_u - 1;

  if(search_range_v_min < 0) search_range_v_min = 0;
  if(search_range_v_max >= num_cell_v) search_range_v_max = num_cell_v - 1;

  for(size_t index_cell_v = search_range_v_min; index_cell_v <= search_range_v_max; ++index_cell_v) {
    const std::vector<std::vector<int>>& reference_grid_row = reference_grid[index_cell_v];
    for(size_t index_cell_u = search_range_u_min; index_cell_u <= search_range_u_max; ++index_cell_u){
      const std::vector<int>& reference_grid_row_column = reference_grid_row[index_cell_u];
      for(const int& index_candidate : reference_grid_row_column){
        candidate_indexes.push_back(index_candidate);
      }
    }
  }
}


int FeatureMatcher::descriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;

	// 총 256 bits.
  for(int i = 0; i < 8; ++i, ++pa, ++pb) {
    unsigned  int v = *pa ^ *pb; // 서로 다르면 1, 같으면 0. 한번에 총 32비트 (4바이트 정수) 
	  // true bit의 갯수를 세는 루틴.
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }
  return dist;
}