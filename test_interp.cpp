#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"

#include "util/timer.h"
#include "util/image_processing.h"

int main()
{
  //  cv::Mat img = cv::imread("Lenna.png", cv::IMREAD_GRAYSCALE);
  cv::Mat cv_image = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);

  const size_t image_width = cv_image.cols;
  const size_t image_height = cv_image.rows;

  std::cout << "Image Size: [" << image_width << ", " << image_height << "]\n";

  // Generate patch. v-major is more faster.
  constexpr float shift = 0.44;
  constexpr int u = 350;
  constexpr int v = 250;
  constexpr int win_size = 21;
  constexpr int max_iter = 100000;
  constexpr int half_win_size = win_size >> 1;
  std::vector<cv::Point2f> pts_sample;
  for (float vv = v - half_win_size; vv <= v + half_win_size; vv += 1)
    for (float uu = u - half_win_size; uu <= u + half_win_size; uu += 1)
      pts_sample.push_back({uu + shift, vv});

  std::vector<std::vector<float>> value_interp(30);
  std::vector<float> value_interp2;
  std::vector<bool> mask_interp;

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::InterpolateImageIntensity(cv_image, pts_sample, value_interp[0], mask_interp);
  std::cout << "InterpolateImageIntensity: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::InterpolateImageIntensitySameRatio(cv_image, pts_sample, shift, 0.0, value_interp[2], mask_interp);
  std::cout << "InterpolateImageIntensitySameRatio: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::InterpolateImageIntensitySameRatioHorizontal(cv_image, pts_sample, shift, value_interp[4], mask_interp);
  std::cout << "InterpolateImageIntensitySameRatioHorizontal: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::unsafe::InterpolateImageIntensity(cv_image, pts_sample, value_interp[1]);
  std::cout << "unsafe::InterpolateImageIntensity: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::unsafe::InterpolateImageIntensitySameRatio(cv_image, pts_sample, shift, 0.0, value_interp[3]);
  std::cout << "unsafe::InterpolateImageIntensitySameRatio: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::unsafe::InterpolateImageIntensitySameRatioHorizontal(cv_image, pts_sample, shift, value_interp[5]);
  std::cout << "unsafe::InterpolateImageIntensitySameRatioHorizontal: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::unsafe::InterpolateImageIntensitySameRatioHorizontalRegularPattern(cv_image, cv::Point2f(u + shift, v), shift, win_size, value_interp[6]);
  std::cout << "unsafe::InterpolateImageIntensitySameRatioHorizontalRegularPattern: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for (int iter = 0; iter < max_iter; ++iter)
    image_processing::unsafe::InterpolateImageIntensitySameRatioHorizontalRegularPatternArbitraryWindow(cv_image, cv::Point2f(u + shift, v), shift, 25, 25, 25, 25, value_interp[7]);
  std::cout << "unsafe::InterpolateImageIntensitySameRatioHorizontalRegularPatternArbitraryWindow: " << timer::toc(0) << " [ms]\n";

  // for(int i = 0; i < value_interp[0].size(); ++i){
  //   for(int j = 0; j < 7; ++j)
  //     std::cout << value_interp[j][i] << " ";

  //   std::cout << "\n";
  // }

  // std::cout << "length of patch : " << pts_sample.size() << std::endl;

  std::vector<unsigned char> values(value_interp[7].size(), 255);
  for (int i = 0; i < value_interp[7].size(); ++i)
    values[i] = (uchar)value_interp[7][i];

  int sz = std::sqrt(pts_sample.size());
  cv::Mat cv_image_interp(sz, sz, CV_8U);
  memcpy(cv_image_interp.ptr<uchar>(0), values.data(), sizeof(unsigned char) * sz * sz);

  // Showing
  cv::Point2f tl(999, 999);
  cv::Point2f br(0, 0);

  for (const auto &pt : pts_sample)
  {
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
