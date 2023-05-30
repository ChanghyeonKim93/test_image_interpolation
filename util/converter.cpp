#include "converter.h"

void converter::convertCvKeyPointToCvPoint(
    const std::vector<cv::KeyPoint>& kpts,
    std::vector<cv::Point2f>& pts)
{
  const size_t n_pts = kpts.size();
  pts.resize(n_pts);

  if(kpts.empty()) return;
  
  for(size_t i = 0; i < n_pts; ++i) {
    pts[i] = kpts[i].pt;
  }  
};