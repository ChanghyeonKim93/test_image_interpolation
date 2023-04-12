#include "klt_tracker.h"

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

void KLTTracker::trackForwardAdditive(const cv::Mat &img0, const cv::Mat &img1,
                                      const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts_track,
                                      const size_t max_level, const size_t window_size, const size_t max_iteration,
                                      const bool use_sparse_patch, const bool use_initial_flow)
{
  if (img0.cols != img1.cols || img0.rows != img1.rows)
  {
    throw std::runtime_error("img0.cols != img1.cols || img0.rows != img1.rows");
  }

  if (!use_initial_flow)
    pts_track = pts0;

  std::vector<cv::Mat> imgpyr0, imgpyr1;
  image_processing::generateImagePyramid(img0, imgpyr0, max_level);
  image_processing::generateImagePyramid(img1, imgpyr1, max_level);

  std::vector<cv::Point2f> patch_pixels;
  this->generatePatchPixels(window_size, patch_pixels, use_sparse_patch);

  const size_t n_pts = pts0.size();
  for (size_t pt_index = 0; pt_index < n_pts; ++pt_index)
  {
    const float scaler = (std::pow(0.5f, max_level - 1));
    std::cout << "scaler: " << scaler << std::endl;

    cv::Point2f pt0 = pts0[pt_index] * scaler;
    cv::Point2f pt_track = pts_track[pt_index] * scaler;

    for (int lvl = max_level - 1; lvl >= 0; --lvl)
    {
      std::cout << "n pt: " << pt_index << ", lvl: " << lvl << "\n";
      std::cout << "pt0, pt_track: " << pt0 << ", " << pt_track << std::endl;

      const cv::Mat &I0 = imgpyr0[lvl];
      const cv::Mat &I1 = imgpyr1[lvl];

      std::vector<float> I0_patch(patch_pixels.size(), 0);
      std::vector<float> I1_patch(patch_pixels.size(), 0);
      std::vector<float> du1_patch(patch_pixels.size(), 0);
      std::vector<float> dv1_patch(patch_pixels.size(), 0);

      std::vector<cv::Point2f> pixels0 = patch_pixels;
      for (cv::Point2f &pt_temp : pixels0)
        pt_temp += pt0;

      image_processing::unsafe::interpImageSameRatio(I0, pixels0,
                                                     pt0.x - floor(pt0.x), pt0.y - floor(pt0.y), I0_patch);
      
                   
      for (size_t iter = 0; iter < max_iteration; ++iter)
      {
        // std::vector<cv::Point2f> pixels0 = patch_pixels;
        // for (cv::Point2f &pt_temp : pixels0)
        //   pt_temp += pt0;
        // image_processing::interpImageSameRatio(I0,)
      }

      pt0 *= 2.0f;
      pt_track *= 2.0f;
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

void KLTTracker::generatePatchPixels(const size_t window_size,
                                     std::vector<cv::Point2f> &patch_pixels,
                                     const bool make_sparse_patch)
{
  if (!(window_size & 0b01))
    throw std::runtime_error("'window_size' should be odd number.");

  int half_window_size = static_cast<int>(window_size * 0.5);
  patch_pixels.reserve(window_size * window_size);
  if (make_sparse_patch)
  {
    for (int v = -half_window_size; v <= half_window_size; ++v) // should be row-major
    {
      int u_add = 0;
      if (v & 0b01)
        u_add = 1;

      for (int u = -half_window_size + u_add; u <= half_window_size; u += 2)
      {
        patch_pixels.emplace_back(u, v);
      }
    }
  }
  else
  {
    for (int v = -half_window_size; v <= half_window_size; ++v) // should be row-major
    {
      for (int u = -half_window_size; u <= half_window_size; ++u)
      {
        patch_pixels.emplace_back(u, v);
      }
    }
  }
}
