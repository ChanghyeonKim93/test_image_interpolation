#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

// References:
//  https://github.com/JiatianWu/stereo-dso/blob/master/src/util/globalFuncs.h#L243
//  https://github.com/uzh-rpg/rpg_svo/blob/d6161063b47f36ce78252ee4c4fedf3f6d8f2898/svo/src/feature_alignment.cpp#L355
#include <iostream>

#include "eigen3/Eigen/Dense"
#include "opencv4/opencv2/core.hpp"

/*
    I00 ax   1-ax I01
    ay   (u,v)

    1-ay
    I10           I11
*/
namespace image_processing {

std::string ConvertImageTypeToString(const cv::Mat &cv_image);

bool IsPixelPositionInImage(const Eigen::Vector2f &pixel_position, const int image_width,
                            const int image_height);
bool IsPixelPositionInImage(const float pixel_u, const float pixel_v, const int image_width,
                            const int image_height);

cv::Mat DownsampleImage(const cv::Mat &cv_source_image);
cv::Mat GeneratePaddedImageByMirroring(const cv::Mat &cv_source_image, const int pad_size);
std::vector<cv::Mat> GenerateImagePyramid(const cv::Mat &cv_source_image, const int num_levels,
                                          const bool use_padding = false, const int pad_size = 0);
// void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
//                           const int max_level, const bool use_padding, const int pad_size,
//                           std::vector<cv::Mat> &pyramid_du, std::vector<cv::Mat> &pyramid_dv);

// clang-format off
// [V] query_pixel : 임의 픽셀
// [V] query_pixel_list : 임의 픽셀 리스트
// [V] (u,v) center_pixel + pattern_pixel_list : 중심과 패치 로컬 패턴 (임의 지정)
// [V] (u,v) center_pixel + horizon_winsz & vertical_winsz 중심과 패치 윈도우 사이즈 (같은 비율) 
// [ ] (u  ) center_pixel + horizon_winsz & vertical_winsz 중심과 패치 윈도우 사이즈 (v 정수)
// [ ] (v  ) center_pixel + horizon_winsz & vertical_winsz 중심과 패치 윈도우 사이즈 (u 정수)
// clang-format on
float InterpolateImageIntensity(const cv::Mat &cv_source_image,
                                const Eigen::Vector2f &pixel_position);
std::vector<float>
  InterpolateImageIntensity(const cv::Mat &cv_source_image,
                            const std::vector<Eigen::Vector2f> &pixel_position_list);
std::vector<float> InterpolateImageIntensityWithPatchPattern(
  const cv::Mat &cv_source_image, const Eigen::Vector2f &patch_center_pixel_position,
  const std::vector<Eigen::Vector2i> &patch_local_pixel_position_list);
std::vector<float>
  InterpolateImageIntensityWithPatchSize(const cv::Mat &cv_source_image,
                                         const Eigen::Vector2f &patch_center_pixel_position,
                                         const int patch_width, const int patch_height);
std::vector<float> InterpolateImageIntensityWithIntegerRow(const cv::Mat &cv_source_image,
                                                           const float patch_center_u,
                                                           const int patch_center_v_floor,
                                                           const int patch_width,
                                                           const int patch_height);
std::vector<float> InterpolateImageIntensityWithIntegerColumn(const cv::Mat &cv_source_image,
                                                              const int patch_center_u_floor,
                                                              const float patch_center_v,
                                                              const int patch_width,
                                                              const int patch_height);

// std::vector<float> InterpolateImageIntensity_SIMD_INTEL();

namespace unsafe {

std::vector<float>
  InterpolateImageIntensity(const cv::Mat &cv_source_image,
                            const std::vector<Eigen::Vector2f> &pixel_position_list);
std::vector<float> InterpolateImageIntensityWithPatchPattern(
  const cv::Mat &cv_source_image, const Eigen::Vector2f &patch_center_pixel_position,
  const std::vector<Eigen::Vector2i> &patch_local_pixel_position_list);
std::vector<float>
  InterpolateImageIntensityWithPatchSize(const cv::Mat &cv_source_image,
                                         const Eigen::Vector2f &patch_center_pixel_position,
                                         const int patch_width, const int patch_height);
std::vector<float> InterpolateImageIntensityWithIntegerRow(const cv::Mat &cv_source_image,
                                                           const float patch_center_u,
                                                           const int patch_center_v_floor,
                                                           const int patch_width,
                                                           const int patch_height);
std::vector<float> InterpolateImageIntensityWithIntegerColumn(const cv::Mat &cv_source_image,
                                                              const int patch_center_u_floor,
                                                              const float patch_center_v,
                                                              const int patch_width,
                                                              const int patch_height);

}; // namespace unsafe
}; // namespace image_processing

#endif