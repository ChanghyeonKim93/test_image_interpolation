#include "util/klt_tracker.h"

KLTTracker::KLTTracker()
{
  this->allocateAlignedMemory();
}

KLTTracker::~KLTTracker()
{
  aligned_memory::free<float>(upattern_dense_);
  aligned_memory::free<float>(vpattern_dense_);

  aligned_memory::free<float>(upattern_sparse_);
  aligned_memory::free<float>(vpattern_sparse_);

  aligned_memory::free<float>(buf_u_);
  aligned_memory::free<float>(buf_v_);
  aligned_memory::free<float>(buf_u_warp_);
  aligned_memory::free<float>(buf_v_warp_);

  aligned_memory::free<float>(buf_I1_);
  aligned_memory::free<float>(buf_du1_);
  aligned_memory::free<float>(buf_dv1_);

  aligned_memory::free<float>(buf_I2_);
  aligned_memory::free<float>(buf_du2_);
  aligned_memory::free<float>(buf_dv2_);

  aligned_memory::free<float>(buf_residual_);
  aligned_memory::free<float>(buf_weight_);

  aligned_memory::free<float>(err_ssd_);
  aligned_memory::free<float>(err_ncc_);
  aligned_memory::free<bool>(mask_);
}

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

void KLTTracker::allocateAlignedMemory()
{
  upattern_dense_ = aligned_memory::malloc<float>(10000);
  vpattern_dense_ = aligned_memory::malloc<float>(10000);

  upattern_sparse_ = aligned_memory::malloc<float>(2500);
  vpattern_sparse_ = aligned_memory::malloc<float>(2500);

  buf_u_ = aligned_memory::malloc<float>(50000);
  buf_v_ = aligned_memory::malloc<float>(50000);
  buf_u_warp_ = aligned_memory::malloc<float>(50000);
  buf_v_warp_ = aligned_memory::malloc<float>(50000);

  buf_I1_ = aligned_memory::malloc<float>(50000);
  buf_du1_ = aligned_memory::malloc<float>(50000);
  buf_dv1_ = aligned_memory::malloc<float>(50000);

  buf_I2_ = aligned_memory::malloc<float>(50000);
  buf_du2_ = aligned_memory::malloc<float>(50000);
  buf_dv2_ = aligned_memory::malloc<float>(50000);

  buf_residual_ = aligned_memory::malloc<float>(50000);
  buf_weight_ = aligned_memory::malloc<float>(50000);
 
  err_ssd_ = aligned_memory::malloc<float>(50000);
  err_ncc_ = aligned_memory::malloc<float>(50000);
  mask_ = aligned_memory::malloc<bool>(50000);
}
