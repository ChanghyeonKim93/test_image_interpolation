#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

// References:
//  https://github.com/JiatianWu/stereo-dso/blob/master/src/util/globalFuncs.h#L243
//  https://github.com/uzh-rpg/rpg_svo/blob/d6161063b47f36ce78252ee4c4fedf3f6d8f2898/svo/src/feature_alignment.cpp#L355
#include <iostream>

#include "eigen3/Eigen/Dense"
#include "opencv4/opencv2/core.hpp"

#include "aligned_memory.h"
#include "simd_library.h"

/*
    I00 ax   1-ax I01
    ay   (u,v)

    1-ay
    I10           I11
*/

class ImageProcessing {
public:
  static std::string ConvertImageTypeToString(const cv::Mat &cv_image);

  template <typename T>
  static cv::Mat GenerateImageByData(const std::vector<T> &image_data, const int image_width,
                                     const int image_height);

  static bool IsPixelPositionInImage(const Eigen::Vector2f &pixel_position, const int image_width,
                                     const int image_height);
  static bool IsPixelPositionInImage(const float pixel_u, const float pixel_v,
                                     const int image_width, const int image_height);

  static float CalculateSumOfSquaredDistance(const std::vector<float> &intensity_list_1,
                                             const std::vector<float> &intensity_list_2);
  static float CalculateSumOfSquaredDistance(const cv::Mat &cv_image_1, const cv::Mat &cv_image_2);

  static cv::Mat DownsampleImage(const cv::Mat &cv_source_image);
  static cv::Mat GeneratePaddedImageByMirroring(const cv::Mat &cv_source_image, const int pad_size);
  static std::vector<cv::Mat> GenerateImagePyramid(const cv::Mat &cv_source_image,
                                                   const int num_levels,
                                                   const bool use_padding = false,
                                                   const int pad_size = 0);
  // void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
  //                           const int max_level, const bool use_padding, const int pad_size,
  //                           std::vector<cv::Mat> &pyramid_du, std::vector<cv::Mat> &pyramid_dv);

public:
  static float InterpolateImageIntensity(const cv::Mat &cv_source_image,
                                         const Eigen::Vector2f &pixel_position);
  static std::vector<float>
    InterpolateImageIntensity(const cv::Mat &cv_source_image,
                              const std::vector<Eigen::Vector2f> &pixel_position_list);
  static std::vector<float> InterpolateImageIntensityWithPatchPattern(
    const cv::Mat &cv_source_image, const Eigen::Vector2f &patch_center_pixel_position,
    const std::vector<Eigen::Vector2i> &patch_local_pixel_position_list);
  static std::vector<float>
    InterpolateImageIntensityWithPatchSize(const cv::Mat &cv_source_image,
                                           const Eigen::Vector2f &patch_center_pixel_position,
                                           const int patch_width, const int patch_height);
  static std::vector<float> InterpolateImageIntensityWithIntegerRow(const cv::Mat &cv_source_image,
                                                                    const float patch_center_u,
                                                                    const int patch_center_v_floor,
                                                                    const int patch_width,
                                                                    const int patch_height);
  static std::vector<float> InterpolateImageIntensityWithIntegerColumn(
    const cv::Mat &cv_source_image, const int patch_center_u_floor, const float patch_center_v,
    const int patch_width, const int patch_height);

  static std::vector<float>
    InterpolateImageIntensity_unsafe(const cv::Mat &cv_source_image,
                                     const std::vector<Eigen::Vector2f> &pixel_position_list);
  static std::vector<float> InterpolateImageIntensityWithPatchPattern_unsafe(
    const cv::Mat &cv_source_image, const Eigen::Vector2f &patch_center_pixel_position,
    const std::vector<Eigen::Vector2i> &patch_local_pixel_position_list);
  static std::vector<float> InterpolateImageIntensityWithPatchSize_unsafe(
    const cv::Mat &cv_source_image, const Eigen::Vector2f &patch_center_pixel_position,
    const int patch_width, const int patch_height);
  static std::vector<float> InterpolateImageIntensityWithIntegerRow_unsafe(
    const cv::Mat &cv_source_image, const float patch_center_u, const int patch_center_v_floor,
    const int patch_width, const int patch_height);
  static std::vector<float> InterpolateImageIntensityWithIntegerColumn_unsafe(
    const cv::Mat &cv_source_image, const int patch_center_u_floor, const float patch_center_v,
    const int patch_width, const int patch_height);

public:
  static std::vector<float> InterpolateImageIntensityWithIntegerRow_simd_intel(
    const cv::Mat &cv_source_image, const float patch_center_u, const int patch_center_v_floor,
    const int patch_width, const int patch_height);

  // private: // for simd
  //   static float *buf_u_;
  //   static float *buf_v_;
  //   static float *buf_I00_;
  //   static float *buf_I01_;
  //   static float *buf_I10_;
  //   static float *buf_I11_;
  //   static float *buf_interped_;
};

template <typename T>
cv::Mat ImageProcessing::GenerateImageByData(const std::vector<T> &image_data,
                                             const int image_width, const int image_height) {
  const int num_data = image_height * image_width;
  if (num_data != image_data.size())
    throw std::runtime_error("num_data != image_data.size()\n");

  std::vector<uint8_t> values(image_data.size(), 0);
  for (int i = 0; i < image_data.size(); ++i)
    values[i] =
      static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(image_data[i])), 0, 255));

  cv::Mat cv_image_interp(image_height, image_width, CV_8UC1);
  memcpy(cv_image_interp.ptr<uint8_t>(0), values.data(), sizeof(uint8_t) * num_data);

  return cv_image_interp;
}

#endif