#include "util/klt_tracker.h"

KLTTracker::KLTTracker(){

};

void KLTTracker::setMaxIteration(const size_t max_iter)
{
  max_iter_ = max_iter;
}

void KLTTracker::track(const cv::Mat &img0, const cv::Mat &img1,
                       const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
                       bool is_use_initial_flow)
{
  if (img0.cols != img1.cols || img0.rows != img1.rows)
  {
    throw std::runtime_error("img0.cols != img1.cols || img0.rows != img1.rows");
  }

  const size_t n_pts = pts0.size();
  if (is_use_initial_flow)
  {
    if (pts0.size() != pts_track.size())
    {
      throw std::runtime_error("pts0.size() != pts_track.size()");
    }
  }
  else
  {
    for (size_t index = 0; index < n_pts; ++index)
    {
      const cv::Point2f &pt0 = pts0[index];
      const cv::Point2f &pt_track = pts_track[index];

    }
  }
}
