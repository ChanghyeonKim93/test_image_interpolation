#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/imgproc.hpp"

#include "util/image_processing.h"
#include "util/timer.h"

int main() {
  //  cv::Mat img = cv::imread("Lenna.png", cv::IMREAD_GRAYSCALE);
  cv::Mat cv_image =
    cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;

  std::cout << "Image Size: [" << image_width << ", " << image_height << "]\n";

  // Generate patch. v-major is more faster.
  constexpr float shift = 0.44f;
  constexpr int u = 350;
  constexpr int v = 250;
  constexpr int win_size = 111;
  constexpr int max_iter = 1000;
  constexpr int half_win_size = win_size >> 1;

  Eigen::Vector2f patch_center_pixel(u + shift, v);

  std::vector<Eigen::Vector2f> pixel_list;
  for (float vv = v - half_win_size; vv <= v + half_win_size; vv += 1)
    for (float uu = u - half_win_size; uu <= u + half_win_size; uu += 1)
      pixel_list.push_back({uu + shift, vv});

  std::vector<Eigen::Vector2i> patch_local_pixel_position_list;
  for (int vv = -half_win_size; vv <= half_win_size; vv += 1)
    for (int uu = -half_win_size; uu <= half_win_size; uu += 1)
      patch_local_pixel_position_list.push_back({uu, vv});

  std::vector<cv::Point2f> pts_sample;
  for (float vv = v - half_win_size; vv <= v + half_win_size; vv += 1)
    for (float uu = u - half_win_size; uu <= u + half_win_size; uu += 1)
      pts_sample.push_back({uu + shift, vv});

  std::vector<std::vector<float>> interp_result_list(30);
  std::vector<float> value_interp2;
  std::vector<bool> mask_interp;

  // Safe
  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[0] = ImageProcessing::InterpolateImageIntensity(cv_image, pixel_list);
  std::cout << "InterpolateImageIntensity: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[1] = ImageProcessing::InterpolateImageIntensityWithPatchPattern(
      cv_image, patch_center_pixel, patch_local_pixel_position_list);
  std::cout << "InterpolateImageIntensityWithPatchPattern: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[2] = ImageProcessing::InterpolateImageIntensityWithPatchSize(
      cv_image, patch_center_pixel, win_size, win_size);
  std::cout << "InterpolateImageIntensityWithPatchSize: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[3] = ImageProcessing::InterpolateImageIntensityWithIntegerRow(
      cv_image, patch_center_pixel.x(), static_cast<int>(patch_center_pixel.y()), win_size,
      win_size);
  std::cout << "InterpolateImageIntensityWithIntegerRow: " << timer::toc(0) << " [ms]\n";

  // Unsafe
  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[4] = ImageProcessing::InterpolateImageIntensity_unsafe(cv_image, pixel_list);
  std::cout << "unsafe::InterpolateImageIntensity: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[5] = ImageProcessing::InterpolateImageIntensityWithPatchPattern_unsafe(
      cv_image, patch_center_pixel, patch_local_pixel_position_list);
  std::cout << "unsafe::InterpolateImageIntensityWithPatchPattern: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[6] = ImageProcessing::InterpolateImageIntensityWithPatchSize_unsafe(
      cv_image, patch_center_pixel, win_size, win_size);
  std::cout << "unsafe::InterpolateImageIntensityWithPatchSize: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    interp_result_list[7] = ImageProcessing::InterpolateImageIntensityWithIntegerRow_unsafe(
      cv_image, patch_center_pixel.x(), static_cast<int>(patch_center_pixel.y()), win_size,
      win_size);
  std::cout << "unsafe::InterpolateImageIntensityWithIntegerRow: " << timer::toc(0) << " [ms]\n";

  // Draw result
  const int test_index = 7;
  std::vector<uint8_t> values(interp_result_list[test_index].size(), 0);
  for (int i = 0; i < interp_result_list[test_index].size(); ++i)
    values[i] = static_cast<uint8_t>(interp_result_list[test_index][i]);

  int sz = std::sqrt(pts_sample.size());
  cv::Mat cv_image_interp(sz, sz, CV_8U);
  memcpy(cv_image_interp.ptr<uint8_t>(0), values.data(), sizeof(uint8_t) * sz * sz);

  // Showing
  cv::Point2f tl(999, 999);
  cv::Point2f br(0, 0);

  for (const auto &pt : pts_sample) {
    if (tl.x > pt.x)
      tl.x = pt.x;
    if (tl.y > pt.y)
      tl.y = pt.y;
    if (br.x < pt.x)
      br.x = pt.x;
    if (br.y < pt.y)
      br.y = pt.y;
  }
  cv::Rect rect(tl, br);

  cv::rectangle(cv_image, rect, cv::Scalar(255, 255, 255), 1);

  cv::namedWindow("cv_image");
  cv::imshow("cv_image", cv_image);

  cv::namedWindow("cv_image_interp");
  cv::imshow("cv_image_interp", cv_image_interp);

  cv::waitKey(0);

  return 1;
};
