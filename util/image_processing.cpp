#include "image_processing.h"

namespace image_processing {
std::string ConvertImageTypeToString(const cv::Mat &cv_image) {
  std::string image_type_string;

  const int cv_image_type = cv_image.type();
  uint8_t depth = cv_image_type & CV_MAT_DEPTH_MASK;
  uint8_t num_channels = 1 + (cv_image_type >> CV_CN_SHIFT);
  switch (depth) {
  case CV_8U:
    image_type_string = "8U";
    break;
  case CV_8S:
    image_type_string = "8S";
    break;
  case CV_16U:
    image_type_string = "16U";
    break;
  case CV_16S:
    image_type_string = "16S";
    break;
  case CV_32S:
    image_type_string = "32S";
    break;
  case CV_32F:
    image_type_string = "32F";
    break;
  case CV_64F:
    image_type_string = "64F";
    break;
  default:
    image_type_string = "User";
    break;
  }
  image_type_string += "C";
  image_type_string += (num_channels + '0');

  return image_type_string;
}

bool IsPixelPositionInImage(const Eigen::Vector2f &pixel_position, const int image_width,
                            const int image_height) {
  const auto &pixel_u = pixel_position.x();
  const auto &pixel_v = pixel_position.y();
  return (pixel_u > 0 && pixel_u <= image_width - 1 && pixel_v > 0 && pixel_v <= image_height - 1);
}

bool IsPixelPositionInImage(const float pixel_u, const float pixel_v, const int image_width,
                            const int image_height) {
  return (pixel_u > 0 && pixel_u <= image_width - 1 && pixel_v > 0 && pixel_v <= image_height - 1);
}

cv::Mat DownsampleImage(const cv::Mat &cv_source_image) {
  if (cv_source_image.type() != CV_8UC1)
    throw std::runtime_error("In function " + std::string(__func__) + ", " +
                             "img_src.type() != CV_8U");

  const int &image_width = cv_source_image.cols;
  const int &image_height = cv_source_image.rows;
  const int half_image_width = image_width >> 1;
  const int half_image_height = image_height >> 1;

  cv::Mat cv_result_image = cv::Mat(half_image_height, half_image_width, CV_8UC1);

  const uint8_t *source_ptr = cv_source_image.ptr<uint8_t>(0);
  const uint8_t *source_I00_ptr = source_ptr;
  const uint8_t *source_I01_ptr = source_ptr + 1;
  const uint8_t *source_I10_ptr = source_ptr + image_width;
  const uint8_t *source_I11_ptr = source_ptr + image_width + 1;
  uint8_t *result_ptr = cv_result_image.ptr<uint8_t>(0);
  const uint8_t *result_ptr_end = result_ptr + half_image_width * half_image_height;
  while (result_ptr != result_ptr_end) {
    const uint8_t *source_image_row_end_ptr = source_I00_ptr + image_width;
    for (; source_I00_ptr != source_image_row_end_ptr; ++result_ptr, source_I00_ptr += 2,
                                                       source_I01_ptr += 2, source_I10_ptr += 2,
                                                       source_I11_ptr += 2) {
      const auto &I00 = *source_I00_ptr;
      const auto &I01 = *source_I01_ptr;
      const auto &I10 = *source_I10_ptr;
      const auto &I11 = *source_I11_ptr;
      *result_ptr = static_cast<uint8_t>(static_cast<uint16_t>(I00 + I01 + I10 + I11) >> 2);
    }
    source_I00_ptr += image_width;
    source_I01_ptr += image_width;
    source_I10_ptr += image_width;
    source_I11_ptr += image_width;
  }

  return cv_result_image;
}

cv::Mat GeneratePaddedImageByMirroring(const cv::Mat &cv_source_image, const int pad_size) {
  const int image_width = cv_source_image.cols;
  const int image_height = cv_source_image.rows;
  const int image_width_padded = image_width + 2 * pad_size;
  const int image_height_padded = image_height + 2 * pad_size;

  cv::Mat cv_result_image =
    cv::Mat::zeros(image_height_padded, image_width_padded, cv_source_image.type());

  const uint8_t *source_ptr = cv_source_image.ptr<uint8_t>(0);
  const uint8_t *source_ptr_end = source_ptr + image_width * image_height;
  uint8_t *result_ptr = cv_result_image.ptr<uint8_t>(0) + pad_size * image_width_padded + pad_size;

  // copy original
  const int size_of_original_row = sizeof(uint8_t) * image_width;
  for (; source_ptr != source_ptr_end;
       source_ptr += image_width, result_ptr += image_width_padded) {
    memcpy(result_ptr, source_ptr, size_of_original_row);
  }

  // mirror columns
  uint8_t *ptr_dst_row = cv_result_image.ptr<uint8_t>(pad_size);
  uint8_t *ptr_dst_row_end = cv_result_image.ptr<uint8_t>(image_height_padded - pad_size);
  for (; ptr_dst_row < ptr_dst_row_end; ptr_dst_row += image_width_padded) {
    source_ptr = ptr_dst_row + pad_size;
    source_ptr_end = ptr_dst_row + 2 * pad_size;
    result_ptr = ptr_dst_row + pad_size - 1;
    for (; source_ptr < source_ptr_end; ++source_ptr, --result_ptr)
      *result_ptr = *source_ptr;

    source_ptr = ptr_dst_row + image_width;
    source_ptr_end = ptr_dst_row + image_width + pad_size;
    result_ptr = ptr_dst_row + image_width_padded - 1;
    for (; source_ptr < source_ptr_end; ++source_ptr, --result_ptr)
      *result_ptr = *source_ptr;
  }

  // upper pad
  const int size_of_padded_row = sizeof(uint8_t) * image_width_padded;
  source_ptr = cv_result_image.ptr<uint8_t>(0) + pad_size * image_width_padded;
  source_ptr_end = cv_result_image.ptr<uint8_t>(0) + 2 * pad_size * image_width_padded;
  result_ptr = cv_result_image.ptr<uint8_t>(0) + (pad_size - 1) * image_width_padded;
  for (; source_ptr != source_ptr_end;
       source_ptr += image_width_padded, result_ptr -= image_width_padded) {
    memcpy(result_ptr, source_ptr, size_of_padded_row);
  }

  // lower pad
  source_ptr =
    cv_result_image.ptr<uint8_t>(0) + (image_height_padded - 2 * pad_size) * image_width_padded;
  source_ptr_end =
    cv_result_image.ptr<uint8_t>(0) + (image_height_padded - pad_size) * image_width_padded;
  result_ptr = cv_result_image.ptr<uint8_t>(0) + (image_height_padded - 1) * image_width_padded;
  for (; source_ptr != source_ptr_end;
       source_ptr += image_width_padded, result_ptr -= image_width_padded) {
    memcpy(result_ptr, source_ptr, size_of_padded_row);
  }

  return cv_result_image;
}

// void pad_image(struct image_t *input, struct image_t *output, uint8_t expand)
// {
//   image_create(output, input->w + 2 * expand, input->h + 2 * expand,
//   input->type);

//   uint8_t *input_buf = (uint8_t *)input->buf;
//   uint8_t *output_buf = (uint8_t *)output->buf;

//   // Skip first `expand` rows, iterate through next input->h rows
//   for (uint16_t i = expand; i != (output->h - expand); i++)
//   {

//     // Mirror first `expand` columns
//     for (uint8_t j = 0; j != expand; j++)
//       output_buf[i * output->w + (expand - 1 - j)] = input_buf[(i - expand) *
//       input->w + j];

//     // Copy corresponding row values from input image
//     memcpy(&output_buf[i * output->w + expand], &input_buf[(i - expand) *
//     input->w], sizeof(uint8_t) * input->w);

//     // Mirror last `expand` columns
//     for (uint8_t j = 0; j != expand; j++)
//       output_buf[i * output->w + output->w - expand + j] = output_buf[i *
//       output->w + output->w - expand - 1 - j];
//   }

//   // Mirror first `expand` and last `expand` rows
//   for (uint8_t i = 0; i != expand; i++)
//   {
//     memcpy(&output_buf[(expand - 1) * output->w - i * output->w],
//     &output_buf[expand * output->w + i * output->w], sizeof(uint8_t) *
//     output->w); memcpy(&output_buf[(output->h - expand) * output->w + i *
//     output->w], &output_buf[(output->h - expand - 1) * output->w - i *
//     output->w], sizeof(uint8_t) * output->w);
//   }
// }

std::vector<cv::Mat> GenerateImagePyramid(const cv::Mat &cv_source_image, const int num_levels,
                                          const bool use_padding, const int pad_size) {
  if (cv_source_image.type() != CV_8U)
    throw std::runtime_error("cv_source_image.type() != CV_8U");

  std::vector<cv::Mat> image_pyramid;
  image_pyramid.resize(num_levels);

  const int &image_width = cv_source_image.cols;
  const int &image_height = cv_source_image.rows;

  cv_source_image.copyTo(image_pyramid[0]);
  for (int lvl = 1; lvl < num_levels; ++lvl) {
    image_pyramid[lvl] = image_processing::DownsampleImage(image_pyramid[lvl - 1]);
  }

  if (use_padding) {
    if (pad_size == 0) {
      throw std::runtime_error("pad_size == 0");
    }
    for (int lvl = num_levels - 1; lvl >= 0; --lvl) {
      image_pyramid[lvl] = GeneratePaddedImageByMirroring(image_pyramid[lvl], pad_size);
    }
  }

  return image_pyramid;
}

// void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
//                           const int max_level, const bool use_padding, const int pad_size,
//                           std::vector<cv::Mat> &pyramid_du, std::vector<cv::Mat> &pyramid_dv) {
//   const int n_cols = img_src.cols;
//   const int n_rows = img_src.rows;

//   pyramid.resize(max_level);
//   img_src.copyTo(pyramid[0]);

//   for (int lvl = 1; lvl < max_level; ++lvl) {
//     const cv::Mat &img_org = pyramid[lvl - 1];
//     cv::Mat &img_dst = pyramid[lvl];
//     image_processing::DownsampleImage(img_org, &img_dst);
//   }

//   if (use_padding) {
//     if (pad_size == 0) {
//       throw std::runtime_error("pad_size == 0");
//     }
//     for (int lvl = max_level - 1; lvl >= 0; --lvl) {
//       cv::Mat img_tmp(pyramid[lvl]);
//       GeneratePaddedImageByMirroring(img_tmp, pyramid[lvl], pad_size);
//     }
//   }

//   pyramid_du.resize(max_level);
//   pyramid_dv.resize(max_level);
//   for (int lvl = 0; lvl < max_level; ++lvl) {
//     pyramid_du[lvl] = cv::Mat(pyramid[lvl].size(), CV_16S);
//     pyramid_dv[lvl] = cv::Mat(pyramid[lvl].size(), CV_16S);
//   }
// }

float InterpolateImageIntensity(const cv::Mat &cv_image, const Eigen::Vector2f &pixel_position) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");

  float interpolated_intensity = -1.0f;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;

  if (!IsPixelPositionInImage(pixel_position, image_width, image_height))
    return -1.0f;

  const float &pixel_u = pixel_position.x();
  const float &pixel_v = pixel_position.y();
  const int pixel_u_floor = static_cast<int>(std::floor(pixel_u));
  const int pixel_v_floor = static_cast<int>(std::floor(pixel_v));

  const float ax = pixel_u - pixel_u_floor;
  const float ay = pixel_v - pixel_v_floor;
  const float axay = ax * ay;

  const int left_upper_index = pixel_v_floor * image_width + pixel_u_floor;
  const uint8_t *image_ptr = cv_image.ptr<uint8_t>(0) + left_upper_index;
  const float &I00 = *image_ptr;
  const float &I01 = *(++image_ptr);
  const float &I11 = *(image_ptr += image_width);
  const float &I10 = *(--image_ptr);
  interpolated_intensity =
    axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

  return interpolated_intensity;
}

std::vector<float>
  InterpolateImageIntensity(const cv::Mat &cv_image,
                            const std::vector<Eigen::Vector2f> &pixel_position_list) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");

  std::vector<float> interpolated_intensity_list;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;
  const uint8_t *source_ptr = cv_image.ptr<uint8_t>(0);

  const size_t num_pixels = pixel_position_list.size();
  interpolated_intensity_list.resize(num_pixels, -1.0f);

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  std::vector<Eigen::Vector2f>::const_iterator itr_pixel = pixel_position_list.begin();
  for (; itr_interpolated_intensity != interpolated_intensity_list.end();
       ++itr_interpolated_intensity, ++itr_pixel) {
    const auto &pixel = *itr_pixel;
    if (!IsPixelPositionInImage(pixel, image_width, image_height))
      continue;

    const auto &u = pixel.x();
    const auto &v = pixel.y();
    const int u_floor = static_cast<int>(std::floor(u));
    const int v_floor = static_cast<int>(std::floor(v));

    const float ax = u - u_floor;
    const float ay = v - v_floor;
    const float axay = ax * ay;

    const uint8_t *source_I00_ptr = source_ptr + v_floor * image_width + u_floor;
    const float &I00 = *source_I00_ptr;
    const float &I01 = *(++source_I00_ptr);
    const float &I11 = *(source_I00_ptr += image_width);
    const float &I10 = *(--source_I00_ptr);

    const float interpolated_intensity =
      axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

    *itr_interpolated_intensity = interpolated_intensity;
  }

  return interpolated_intensity_list;
}

std::vector<float> InterpolateImageIntensityWithPatchPattern(
  const cv::Mat &cv_image, const Eigen::Vector2f &patch_center_pixel_position,
  const std::vector<Eigen::Vector2i> &patch_local_pixel_position_list) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");

  std::vector<float> interpolated_intensity_list;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;
  const uint8_t *source_ptr = cv_image.ptr<uint8_t>(0);

  const size_t num_patch_pixels = patch_local_pixel_position_list.size();
  interpolated_intensity_list.resize(num_patch_pixels, -1.0f);

  const auto &u_center = patch_center_pixel_position.x();
  const auto &v_center = patch_center_pixel_position.y();
  const int u_center_floor = static_cast<int>(std::floor(u_center));
  const int v_center_floor = static_cast<int>(std::floor(v_center));
  const float ax = u_center - u_center_floor;
  const float ay = v_center - v_center_floor;
  const float axay = ax * ay;

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  std::vector<Eigen::Vector2i>::const_iterator itr_pixel = patch_local_pixel_position_list.begin();
  for (; itr_interpolated_intensity != interpolated_intensity_list.end();
       ++itr_interpolated_intensity, ++itr_pixel) {
    const auto &local_patch_pixel = *itr_pixel;
    const int u_floor = u_center_floor + local_patch_pixel.x();
    const int v_floor = v_center_floor + local_patch_pixel.y();
    if (!IsPixelPositionInImage(u_floor, v_floor, image_width, image_height))
      continue;

    const uint8_t *source_I00_ptr = source_ptr + v_floor * image_width + u_floor;
    const float &I00 = *source_I00_ptr;
    const float &I01 = *(++source_I00_ptr);
    const float &I11 = *(source_I00_ptr += image_width);
    const float &I10 = *(--source_I00_ptr);

    const float interpolated_intensity =
      axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

    *itr_interpolated_intensity = interpolated_intensity;
  }

  return interpolated_intensity_list;
}

std::vector<float>
  InterpolateImageIntensityWithPatchSize(const cv::Mat &cv_image,
                                         const Eigen::Vector2f &patch_center_pixel_position,
                                         const int patch_width, const int patch_height) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");
  auto is_odd_number = [](const int number) -> bool { return (number & 0b01); };
  if (!is_odd_number(patch_width))
    throw std::runtime_error("patch_width is not odd number.\n");
  if (!is_odd_number(patch_height))
    throw std::runtime_error("patch_height is not odd number.\n");
  const int half_patch_width = patch_width >> 1;
  const int half_patch_height = patch_height >> 1;

  std::vector<float> interpolated_intensity_list;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;
  const uint8_t *source_ptr = cv_image.ptr<uint8_t>(0);

  const int num_patch_pixels = patch_width * patch_height;
  interpolated_intensity_list.resize(num_patch_pixels, -1.0f);

  const auto &u_center = patch_center_pixel_position.x();
  const auto &v_center = patch_center_pixel_position.y();
  const int u_center_floor = static_cast<int>(std::floor(u_center));
  const int v_center_floor = static_cast<int>(std::floor(v_center));
  const float ax = u_center - u_center_floor;
  const float ay = v_center - v_center_floor;
  const float axay = ax * ay;

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  for (int v = v_center_floor - half_patch_height; v <= v_center_floor + half_patch_height; ++v) {
    const uint8_t *source_row_ptr = source_ptr + v * image_width;
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      if (!IsPixelPositionInImage(u, v, image_width, image_height))
        continue;

      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);
      const float &I11 = *(source_I00_ptr += image_width);
      const float &I10 = *(--source_I00_ptr);

      const float interpolated_intensity =
        axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

      *itr_interpolated_intensity = interpolated_intensity;
    }
  }

  return interpolated_intensity_list;
}

std::vector<float> InterpolateImageIntensityWithIntegerRow(const cv::Mat &cv_image,
                                                           const float patch_center_u,
                                                           const int patch_center_v_floor,
                                                           const int patch_width,
                                                           const int patch_height) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");
  auto is_odd_number = [](const int number) -> bool { return (number & 0b01); };
  if (!is_odd_number(patch_width))
    throw std::runtime_error("patch_width is not odd number.\n");
  if (!is_odd_number(patch_height))
    throw std::runtime_error("patch_height is not odd number.\n");
  const int half_patch_width = patch_width >> 1;
  const int half_patch_height = patch_height >> 1;

  std::vector<float> interpolated_intensity_list;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;
  const uint8_t *source_ptr = cv_image.ptr<uint8_t>(0);

  const int num_patch_pixels = patch_width * patch_height;
  interpolated_intensity_list.resize(num_patch_pixels, -1.0f);

  const auto &u_center = patch_center_u;
  const auto &v_center = patch_center_v_floor;
  const int u_center_floor = static_cast<int>(std::floor(u_center));
  const int v_center_floor = v_center;
  const float ax = u_center - u_center_floor;

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  for (int v = v_center_floor - half_patch_height; v <= v_center_floor + half_patch_height; ++v) {
    const uint8_t *source_row_ptr = source_ptr + v * image_width;
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      if (!IsPixelPositionInImage(u, v, image_width, image_height))
        continue;

      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);

      const float interpolated_intensity = ax * (-I00 + I01);

      *itr_interpolated_intensity = interpolated_intensity;
    }
  }

  return interpolated_intensity_list;
}

// void InterpolateImageIntensitySameRatio(const cv::Mat &cv_source_image,
//                                         const std::vector<cv::Point2f> &pts, float ax, float ay,
//                                         std::vector<float> *value_interp,
//                                         std::vector<bool> *mask_interp) {
//   if (cv_source_image.type() != CV_8U)
//     std::runtime_error("cv_image.type() != CV_8U");

//   const int &image_width = cv_source_image.cols;
//   const int &image_height = cv_source_image.rows;
//   const uint8_t *source_ptr = cv_source_image.ptr<uint8_t>(0);

//   const size_t num_pixels = pts.size();
//   value_interp.resize(num_pixels, -1.0f);
//   mask_interp.resize(num_pixels, false);

//   float axay = ax * ay;

//   std::vector<float>::iterator it_value = value_interp.begin();
//   std::vector<bool>::iterator it_mask = mask_interp.begin();
//   std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
//   for (; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt) {
//     const cv::Point2f &pt = *it_pt;
//     if (pt.x < 0 || pt.x > image_width - 1 || pt.y < 0 || pt.y > image_height - 1)
//       continue;

//     const float uc = pt.x;
//     const float vc = pt.y;
//     int u0 = static_cast<int>(pt.x);
//     int v0 = static_cast<int>(pt.y);

//     int idx_I1 = v0 * image_width + u0;

//     const uint8_t *ptr = source_ptr + idx_I1;
//     const float &I1 = *ptr;                  // v_0n_colsu_0
//     const float &I2 = *(++ptr);              // v_0n_colsu_0 + 1
//     const float &I4 = *(ptr += image_width); // v_0n_colsu_0 + 1 + n_cols
//     const float &I3 = *(--ptr);              // v_0n_colsu_0 + n_cols

//     float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

//     *it_value = I_interp;
//     *it_mask = true;
//   }
// }

// void InterpolateImageIntensitySameRatioHorizontal(const cv::Mat &cv_image,
//                                                   const std::vector<cv::Point2f> &pts, float ax,
//                                                   std::vector<float> &value_interp,
//                                                   std::vector<bool> &mask_interp) {
//   if (cv_image.type() != CV_8U)
//     std::runtime_error("cv_image.type() != CV_8U");

//   const int n_cols = cv_image.cols;
//   const int n_rows = cv_image.rows;
//   const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

//   const int n_pts = pts.size();
//   value_interp.resize(n_pts, -1.0);
//   mask_interp.resize(n_pts, false);

//   std::vector<float>::iterator it_value = value_interp.begin();
//   std::vector<bool>::iterator it_mask = mask_interp.begin();
//   std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

//   for (; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt) {
//     const cv::Point2f &pt = *it_pt;
//     if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
//       continue;

//     int u0 = static_cast<int>(pt.x);
//     int v0 = static_cast<int>(pt.y);

//     int idx_I1 = v0 * n_cols + u0;

//     const uint8_t *ptr = ptr_img + idx_I1;
//     const float &I1 = *ptr;     // v_0n_colsu_0
//     const float &I2 = *(++ptr); // v_0n_colsu_0 + 1

//     float I_interp = (I2 - I1) * ax + I1;

//     *it_value = I_interp;
//     *it_mask = true;
//   }
// }

// void InterpolateImageIntensitySameRatioRegularPattern(const cv::Mat &cv_image,
//                                                       const cv::Point2f &pt_center, int win_size,
//                                                       std::vector<float> &value_interp,
//                                                       std::vector<bool> &mask_interp) {
//   /*
//   data order
//      0  1  2  3  4  5  6
//      7  8  9 10 11 12 13
//     14 15 16 17 18 19 20
//     21 22 23 24 25 26 27
//     28 29 30 31 32 33 34
//     35 36 37 38 39 40 41
//     42 43 44 45 46 47 48
//   */

//   /*
//     I1 I2 I3 I4 I5
//     J1 J2 J3 J4 J5
//     K1 K2 K3 K4 K5

//   */

//   const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));
//   const float ax = pt_center.x - pt_center0.x;
//   const float ay = pt_center.y - pt_center0.y;
//   const float axay = ax * ay;
// }

}; // namespace image_processing

namespace image_processing {
namespace unsafe {

void InterpolateImageIntensitySameRatioHorizontalRegularPattern(const cv::Mat &cv_image,
                                                                const cv::Point2f &pt_center,
                                                                float ax, int win_size,
                                                                std::vector<float> &value_interp) {
  /*
  data order
     0  1  2  3  4  5  6
     7  8  9 10 11 12 13
    14 15 16 17 18 19 20
    21 22 23 24 25 26 27
    28 29 30 31 32 33 34
    35 36 37 38 39 40 41
    42 43 44 45 46 47 48
  */
  /*
    I1 I2 I3 I4 I5
    J1 J2 J3 J4 J5
    K1 K2 K3 K4 K5

    Ia = I2-Ib; // I2*(1-ax)


  // simplified algorithm
    I1 = *ptr;
    Ia = I1*(1-ax);

    for{
      I2 = *(++ptr);
      Ib = I2*ax;
      I_interp = Ia + Ib; // ( == I1*(1-ax) + I2*ax )
      Ia = I2 - Ib; // == I2*(1-ax)
    }
  */
  if (cv_image.type() != CV_8U)
    std::runtime_error("cv_image.type() != CV_8U");
  if (!(win_size & 0x01))
    std::runtime_error("'win_size' should be an odd number!");

  const double one_minus_ax = 1.0 - ax;
  const int n_cols = cv_image.cols;
  const int n_rows = cv_image.rows;
  const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

  const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

  const int half_win_size = static_cast<int>(floor(win_size * 0.5));

  const int n_pts = win_size * win_size;
  value_interp.resize(n_pts, -1.0);

  const uint8_t *ptr_row_start =
    ptr_img + (pt_center0.y - half_win_size) * n_cols + pt_center0.x - half_win_size;
  const uint8_t *ptr_row_end = ptr_row_start + win_size + 2; // TODO: Patch가 화면 밖으로 나갈때!
  const uint8_t *ptr_row_final = ptr_row_start + (win_size)*n_cols;

  int counter = 0;
  std::vector<float>::iterator it_value = value_interp.begin();
  for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols) {
    const uint8_t *ptr = ptr_row_start;
    // const float I1 = *ptr;
    // float I2 = *(++ptr);
    // float Ia = I1 * one_minus_ax;
    // ++ptr;
    // for (; ptr != ptr_row_end; ++ptr) {
    //   float Ib = I2 * ax;
    //   float I_interp = Ia + Ib;

    //   *(it_value) = I_interp;

    //   Ia = I2 - Ib;
    //   I2 = *(ptr);
    //   ++it_value;
    //   ++counter;
    // }
    const float I1 = *ptr;
    float Ia = I1 * (1.0f - ax);
    for (; ptr != ptr_row_end; ++ptr) {
      const float I2 = *(++ptr);
      const float Ib = I2 * ax;
      *(it_value) = Ia + Ib; // ( == I1*(1-ax) + I2*ax )
      Ia = I2 - Ib;          // == I2*(1-ax)
      ++it_value;
      ++counter;
    }
  }
};

void InterpolateImageIntensitySameRatioHorizontalRegularPatternArbitraryWindow(
  const cv::Mat &cv_image, const cv::Point2f &pt_center, float ax, int half_left, int half_right,
  int half_up, int half_down, std::vector<float> &value_interp) {
  /*
  data order
     0  1  2  3  4  5  6
     7  8  9 10 11 12 13
    14 15 16 17 18 19 20
    21 22 23 24 25 26 27
    28 29 30 31 32 33 34
    35 36 37 38 39 40 41
    42 43 44 45 46 47 48
  */
  /*
    I1 I2 I3 I4 I5
    J1 J2 J3 J4 J5
    K1 K2 K3 K4 K5

  // Initial
    Ia = I1*(1-ax);
    Ib = I2*ax;
    I_interp = Ia + Ib; ( == I1*(1-ax) + I2*ax )

  // Consecutive
    Ia = I2 - Ib;
    Ib = I3*ax;
    I_interp = Ia + Ib; ( == I2*(1-ax) + I3*ax ) ...
  */
  if (cv_image.type() != CV_8U)
    std::runtime_error("cv_image.type() != CV_8U");

  const double one_minus_ax = 1.0 - ax;
  const int n_cols = cv_image.cols;
  const int n_rows = cv_image.rows;
  const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

  const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

  const int win_size_horizontal = half_right + half_left + 1;
  const int win_size_vertical = half_down + half_up + 1;

  const int n_pts = win_size_horizontal * win_size_vertical;
  value_interp.resize(n_pts, -1.0);

  const uint8_t *ptr_row_start =
    ptr_img + (pt_center0.y - half_up) * n_cols + pt_center0.x - half_left;
  const uint8_t *ptr_row_end =
    ptr_row_start + win_size_horizontal + 2; // TODO: Patch가 화면 밖으로 나갈때!
  const uint8_t *ptr_row_final = ptr_row_start + (win_size_vertical)*n_cols;

  int counter = 0;
  std::vector<float>::iterator it_value = value_interp.begin();
  for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols) {
    const uint8_t *ptr = ptr_row_start;
    float I1 = *ptr;
    float I2 = *(++ptr);
    float Ia = I1 * one_minus_ax;

    ++ptr;
    for (; ptr != ptr_row_end; ++ptr) {
      float Ib = I2 * ax;
      float I_interp = Ia + Ib;

      *(it_value) = I_interp;

      Ia = I2 - Ib;
      I2 = *(ptr);
      ++it_value;
      ++counter;
    }
  }
};

void InterpolateImageIntensity(const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
                               std::vector<float> &value_interp) {
  if (cv_image.type() != CV_8U)
    std::runtime_error("cv_image.type() != CV_8U");

  const int n_cols = cv_image.cols;
  const int n_rows = cv_image.rows;
  const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

  const int n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for (; it_value != value_interp.end(); ++it_value, ++it_pt) {
    const cv::Point2f &pt = *it_pt;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = static_cast<int>(pt.x);
    int v0 = static_cast<int>(pt.y);

    float ax = uc - u0;
    float ay = vc - v0;
    float axay = ax * ay;
    int idx_I1 = v0 * n_cols + u0;

    const uint8_t *ptr = ptr_img + idx_I1;
    const float &I1 = *ptr;             // v_0n_colsu_0
    const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

    *it_value = I_interp;
  }
};

void InterpolateImageIntensitySameRatio(const cv::Mat &cv_image,
                                        const std::vector<cv::Point2f> &pts, float ax, float ay,
                                        std::vector<float> &value_interp) {
  if (cv_image.type() != CV_8U)
    std::runtime_error("cv_image.type() != CV_8U");

  const int n_cols = cv_image.cols;
  const int n_rows = cv_image.rows;
  const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

  const int n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  float axay = ax * ay;

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for (; it_value != value_interp.end(); ++it_value, ++it_pt) {
    const cv::Point2f &pt = *it_pt;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = static_cast<int>(pt.x);
    int v0 = static_cast<int>(pt.y);

    int idx_I1 = v0 * n_cols + u0;

    const uint8_t *ptr = ptr_img + idx_I1;
    const float &I1 = *ptr;             // v_0n_colsu_0
    const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

    *it_value = I_interp;
  }
};

void InterpolateImageIntensitySameRatioHorizontal(const cv::Mat &cv_image,
                                                  const std::vector<cv::Point2f> &pts, float ax,
                                                  std::vector<float> &value_interp) {
  if (cv_image.type() != CV_8U)
    std::runtime_error("cv_image.type() != CV_8U");

  const int n_cols = cv_image.cols;
  const int n_rows = cv_image.rows;
  const uint8_t *ptr_img = cv_image.ptr<uint8_t>(0);

  const int n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

  for (; it_value != value_interp.end(); ++it_value, ++it_pt) {
    const cv::Point2f &pt = *it_pt;

    int u0 = static_cast<int>(pt.x);
    int v0 = static_cast<int>(pt.y);

    int idx_I1 = v0 * n_cols + u0;

    const uint8_t *ptr = ptr_img + idx_I1;
    const float &I1 = *ptr;     // v_0n_colsu_0
    const float &I2 = *(++ptr); // v_0n_colsu_0 + 1

    float I_interp = (I2 - I1) * ax + I1;

    *it_value = I_interp;
  }
}
}; // namespace unsafe
}; // namespace image_processing