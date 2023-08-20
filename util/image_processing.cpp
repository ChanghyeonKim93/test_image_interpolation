#include "image_processing.h"

std::string ImageProcessing::ConvertImageTypeToString(const cv::Mat &cv_image) {
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

bool ImageProcessing::IsPixelPositionInImage(const Eigen::Vector2f &pixel_position,
                                             const int image_width, const int image_height) {
  const auto &pixel_u = pixel_position.x();
  const auto &pixel_v = pixel_position.y();
  return (pixel_u > 0 && pixel_u <= image_width - 1 && pixel_v > 0 && pixel_v <= image_height - 1);
}

bool ImageProcessing::IsPixelPositionInImage(const float pixel_u, const float pixel_v,
                                             const int image_width, const int image_height) {
  return (pixel_u > 0 && pixel_u <= image_width - 1 && pixel_v > 0 && pixel_v <= image_height - 1);
}

float ImageProcessing::CalculateSumOfSquaredDistance(const std::vector<float> &intensity_list_1,
                                                     const std::vector<float> &intensity_list_2) {
  if (intensity_list_1.size() != intensity_list_2.size())
    throw std::runtime_error("intensity_list_1.size() != intensity_list_2.size()");

  float ssd = 0;
  const size_t num_elements = intensity_list_1.size();
  for (size_t index = 0; index < num_elements; ++index) {
    const float diff = intensity_list_1[index] - intensity_list_2[index];
    ssd += diff * diff;
  }
  ssd /= static_cast<float>(num_elements);

  return ssd;
}

float ImageProcessing::CalculateSumOfSquaredDistance(const cv::Mat &cv_image_1,
                                                     const cv::Mat &cv_image_2) {
  float ssd = 0;
  if (cv_image_1.cols != cv_image_2.cols || cv_image_1.rows != cv_image_2.rows)
    throw std::runtime_error(
      "cv_image_1.cols != cv_image_2.cols || cv_image_1.rows != cv_image_2.rows");

  if (cv_image_1.type() != CV_8UC1 || cv_image_2.type() != CV_8UC1)
    throw std::runtime_error("cv_image_1.type() != CV_8UC1 || cv_image_2.type() != CV_8UC1\n");

  const int &image_width = cv_image_1.cols;
  const int &image_height = cv_image_1.rows;
  const int num_elements = image_width * image_height;

  const uint8_t *ptr1 = cv_image_1.ptr<uint8_t>(0);
  const uint8_t *ptr2 = cv_image_2.ptr<uint8_t>(0);
  const uint8_t *ptr1_end = ptr1 + num_elements + 1;
  for (; ptr1 != ptr1_end; ++ptr1, ++ptr2) {
    const float diff = static_cast<float>(*(ptr1)) - static_cast<float>(*(ptr2));
    ssd += diff * diff;
  }
  ssd /= static_cast<float>(num_elements);

  return ssd;
}

cv::Mat ImageProcessing::DownsampleImage(const cv::Mat &cv_source_image) {
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

cv::Mat ImageProcessing::GeneratePaddedImageByMirroring(const cv::Mat &cv_source_image,
                                                        const int pad_size) {
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

std::vector<cv::Mat> ImageProcessing::GenerateImagePyramid(const cv::Mat &cv_source_image,
                                                           const int num_levels,
                                                           const bool use_padding,
                                                           const int pad_size) {
  if (cv_source_image.type() != CV_8U)
    throw std::runtime_error("cv_source_image.type() != CV_8U");

  std::vector<cv::Mat> image_pyramid;
  image_pyramid.resize(num_levels);

  const int &image_width = cv_source_image.cols;
  const int &image_height = cv_source_image.rows;

  cv_source_image.copyTo(image_pyramid[0]);
  for (int lvl = 1; lvl < num_levels; ++lvl) {
    image_pyramid[lvl] = ImageProcessing::DownsampleImage(image_pyramid[lvl - 1]);
  }

  if (use_padding) {
    if (pad_size == 0) {
      throw std::runtime_error("pad_size == 0");
    }
    for (int lvl = num_levels - 1; lvl >= 0; --lvl) {
      image_pyramid[lvl] =
        ImageProcessing::GeneratePaddedImageByMirroring(image_pyramid[lvl], pad_size);
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

float ImageProcessing::InterpolateImageIntensity(const cv::Mat &cv_image,
                                                 const Eigen::Vector2f &pixel_position) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");

  float interpolated_intensity = -1.0f;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;

  if (!ImageProcessing::IsPixelPositionInImage(pixel_position, image_width, image_height))
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

std::vector<float> ImageProcessing::InterpolateImageIntensity(
  const cv::Mat &cv_image, const std::vector<Eigen::Vector2f> &pixel_position_list) {
  if (cv_image.type() != CV_8U)
    throw std::runtime_error("cv_image.type() != CV_8U");

  std::vector<float> interpolated_intensity_list;

  const int &image_width = cv_image.cols;
  const int &image_height = cv_image.rows;
  const uint8_t *source_ptr = cv_image.ptr<uint8_t>(0);

  const size_t num_pixels = pixel_position_list.size();
  interpolated_intensity_list.resize(num_pixels, -1.0f);

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  int count = 0;
  std::vector<Eigen::Vector2f>::const_iterator itr_pixel = pixel_position_list.begin();
  for (; itr_interpolated_intensity != interpolated_intensity_list.end();
       ++itr_interpolated_intensity, ++itr_pixel) {
    const auto &pixel = *itr_pixel;
    if (!ImageProcessing::IsPixelPositionInImage(pixel, image_width, image_height))
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

std::vector<float> ImageProcessing::InterpolateImageIntensityWithPatchPattern(
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
    if (!ImageProcessing::IsPixelPositionInImage(u_floor, v_floor, image_width, image_height))
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

std::vector<float> ImageProcessing::InterpolateImageIntensityWithPatchSize(
  const cv::Mat &cv_image, const Eigen::Vector2f &patch_center_pixel_position,
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
      if (!ImageProcessing::IsPixelPositionInImage(u, v, image_width, image_height))
        continue;

      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);
      const float &I11 = *(source_I00_ptr += image_width);
      const float &I10 = *(--source_I00_ptr);

      const float interpolated_intensity =
        axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

      *itr_interpolated_intensity = interpolated_intensity;
      ++itr_interpolated_intensity;
    }
  }

  return interpolated_intensity_list;
}

std::vector<float> ImageProcessing::InterpolateImageIntensityWithIntegerRow(
  const cv::Mat &cv_image, const float patch_center_u, const int patch_center_v_floor,
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

  const auto &u_center = patch_center_u;
  const auto &v_center = patch_center_v_floor;
  const int u_center_floor = static_cast<int>(std::floor(u_center));
  const int v_center_floor = v_center;
  const float ax = u_center - u_center_floor;

  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  for (int v = v_center_floor - half_patch_height; v <= v_center_floor + half_patch_height; ++v) {
    const uint8_t *source_row_ptr = source_ptr + v * image_width;
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      if (!ImageProcessing::IsPixelPositionInImage(u, v, image_width, image_height))
        continue;

      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);

      const float interpolated_intensity = I00 + ax * (I01 - I00);

      *itr_interpolated_intensity = interpolated_intensity;
      ++itr_interpolated_intensity;
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

std::vector<float> ImageProcessing::InterpolateImageIntensity_unsafe(
  const cv::Mat &cv_image, const std::vector<Eigen::Vector2f> &pixel_position_list) {
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

std::vector<float> ImageProcessing::InterpolateImageIntensityWithPatchPattern_unsafe(
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

std::vector<float> ImageProcessing::InterpolateImageIntensityWithPatchSize_unsafe(
  const cv::Mat &cv_image, const Eigen::Vector2f &patch_center_pixel_position,
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
  const uint8_t *source_row_ptr = source_ptr + (v_center_floor - half_patch_height) * image_width;
  for (int v = v_center_floor - half_patch_height; v <= v_center_floor + half_patch_height; ++v) {
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);
      const float &I11 = *(source_I00_ptr += image_width);
      const float &I10 = *(--source_I00_ptr);

      const float interpolated_intensity =
        axay * (I00 - I01 - I10 + I11) + ax * (-I00 + I01) + ay * (-I00 + I10) + I00;

      *itr_interpolated_intensity = interpolated_intensity;
      ++itr_interpolated_intensity;
    }
    source_row_ptr += image_width;
  }

  return interpolated_intensity_list;
}

std::vector<float> ImageProcessing::InterpolateImageIntensityWithIntegerRow_unsafe(
  const cv::Mat &cv_image, const float patch_center_u, const int patch_center_v_floor,
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

  const auto &u_center = patch_center_u;
  const auto &v_center = patch_center_v_floor;
  const int u_center_floor = static_cast<int>(std::floor(u_center));
  const int v_center_floor = v_center;
  const float ax = u_center - u_center_floor;

  const uint8_t *source_row_ptr = source_ptr + (v_center_floor - half_patch_height) * image_width;
  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  for (int v = -half_patch_height; v <= half_patch_height; ++v) {
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);

      const float interpolated_intensity = I00 + ax * (I01 - I00);

      *itr_interpolated_intensity = interpolated_intensity;
      ++itr_interpolated_intensity;
    }
    source_row_ptr += image_width;
  }

  return interpolated_intensity_list;
}
/*
  const float I1 = *ptr;
  float Ia = I1 * (1.0f - ax);
  for (; ptr != ptr_row_end; ++ptr) {
    const float I2 = *(++ptr);
    const float Ib = I2 * ax;
    const float I_interp = Ia + Ib; // ( == I1*(1-ax) + I2*ax )
    *(it_value) = I_interp;
    Ia = I2 - Ib; // == I2*(1-ax)
    ++it_value;
    ++counter;
  }
*/

std::vector<float> ImageProcessing::InterpolateImageIntensityWithIntegerRow_simd_intel(
  const cv::Mat &cv_image, const float patch_center_u, const int patch_center_v_floor,
  const int patch_width, const int patch_height) {
  // static float *buf_u_ = nullptr, *buf_v_ = nullptr, *buf_I00_ = nullptr, *buf_I01_ = nullptr,
  //              *buf_I10_ = nullptr, *buf_I11_ = nullptr, *buf_interped_ = nullptr;
  // if (buf_u_ == nullptr) {
  //   buf_u_ = aligned_memory::malloc<float>(10000);
  //   buf_v_ = aligned_memory::malloc<float>(10000);
  //   buf_I00_ = aligned_memory::malloc<float>(10000);
  //   buf_I01_ = aligned_memory::malloc<float>(10000);
  //   buf_I10_ = aligned_memory::malloc<float>(10000);
  //   buf_I11_ = aligned_memory::malloc<float>(10000);
  //   buf_interped_ = aligned_memory::malloc<float>(10000);
  // }

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

  const uint8_t *source_row_ptr = source_ptr + (v_center_floor - half_patch_height) * image_width;
  std::vector<float>::iterator itr_interpolated_intensity = interpolated_intensity_list.begin();
  for (int v = -half_patch_height; v <= half_patch_height; ++v) {
    for (int u = u_center_floor - half_patch_width; u <= u_center_floor + half_patch_width; ++u) {
      const uint8_t *source_I00_ptr = source_row_ptr + u;
      const float &I00 = *source_I00_ptr;
      const float &I01 = *(++source_I00_ptr);

      const float interpolated_intensity = I00 + ax * (I01 - I00);

      *itr_interpolated_intensity = interpolated_intensity;
      ++itr_interpolated_intensity;
    }
    source_row_ptr += image_width;
  }

  return interpolated_intensity_list;
}
