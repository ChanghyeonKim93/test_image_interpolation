#include "image_processing.h"

namespace image_processing
{
  std::string ConvertImageTypeToString(const cv::Mat &img)
  {
    std::string r;
    int type = img.type();

    uint8_t depth = type & CV_MAT_DEPTH_MASK;
    uint8_t chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
    }

    r += "C";
    r += (chans + '0');

    return r;
  }

  float InterpolateImageIntensity(const cv::Mat &img, const cv::Point2f &pt)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const int &n_cols = img.cols;
    const int &n_rows = img.rows;
    const uint8_t *ptr_img = img.ptr<uint8_t>(0);

    if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
      return -1.0f;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = static_cast<int>(std::floor(pt.x));
    int v0 = static_cast<int>(std::floor(pt.y));

    float ax = uc - u0;
    float ay = vc - v0;
    float axay = ax * ay;
    int idx_I1 = v0 * n_cols + u0;

    const uint8_t *ptr = ptr_img + idx_I1;
    const float &I1 = *ptr;             // v_0n_colsu_0
    const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float value_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

    return value_interp;
  }

  void InterpolateImageIntensity(const cv::Mat &img, const std::vector<cv::Point2f> &pts,
                                 std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const int n_cols = img.cols;
    const int n_rows = img.rows;
    const uint8_t *ptr_img = img.ptr<uint8_t>(0);

    const int n_pts = pts.size();
    value_interp.resize(n_pts, -1.0);
    mask_interp.resize(n_pts, false);

    std::vector<float>::iterator it_value = value_interp.begin();
    std::vector<bool>::iterator it_mask = mask_interp.begin();
    std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
    for (; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
    {
      const cv::Point2f &pt = *it_pt;
      if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
        continue;

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
      *it_mask = true;
    }
  }

  void InterpolateImageIntensitySameRatio(
      const cv::Mat &img, const std::vector<cv::Point2f> &pts,
      float ax, float ay,
      std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const int n_cols = img.cols;
    const int n_rows = img.rows;
    const uint8_t *ptr_img = img.ptr<uint8_t>(0);

    const int n_pts = pts.size();
    value_interp.resize(n_pts, -1.0);
    mask_interp.resize(n_pts, false);

    float axay = ax * ay;

    std::vector<float>::iterator it_value = value_interp.begin();
    std::vector<bool>::iterator it_mask = mask_interp.begin();
    std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
    for (; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
    {
      const cv::Point2f &pt = *it_pt;
      if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
        continue;

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
      *it_mask = true;
    }
  }

  void InterpolateImageIntensitySameRatioHorizontal(
      const cv::Mat &img, const std::vector<cv::Point2f> &pts,
      float ax,
      std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const int n_cols = img.cols;
    const int n_rows = img.rows;
    const uint8_t *ptr_img = img.ptr<uint8_t>(0);

    const int n_pts = pts.size();
    value_interp.resize(n_pts, -1.0);
    mask_interp.resize(n_pts, false);

    std::vector<float>::iterator it_value = value_interp.begin();
    std::vector<bool>::iterator it_mask = mask_interp.begin();
    std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

    for (; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
    {
      const cv::Point2f &pt = *it_pt;
      if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
        continue;

      int u0 = static_cast<int>(pt.x);
      int v0 = static_cast<int>(pt.y);

      int idx_I1 = v0 * n_cols + u0;

      const uint8_t *ptr = ptr_img + idx_I1;
      const float &I1 = *ptr;     // v_0n_colsu_0
      const float &I2 = *(++ptr); // v_0n_colsu_0 + 1

      float I_interp = (I2 - I1) * ax + I1;

      *it_value = I_interp;
      *it_mask = true;
    }
  }

  void InterpolateImageIntensitySameRatioRegularPattern(
      const cv::Mat &img, const cv::Point2f &pt_center, int win_size,
      std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
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

    */

    const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));
    const float ax = pt_center.x - pt_center0.x;
    const float ay = pt_center.y - pt_center0.y;
    const float axay = ax * ay;
  }

  void DownsampleImage(const cv::Mat &img_src, cv::Mat &img_dst)
  {
    if (img_src.type() != CV_8U)
      throw std::runtime_error("In function " + std::string(__func__) + ", " + "img_src.type() != CV_8U");

    const int n_cols = img_src.cols;
    const int n_rows = img_src.rows;

    const int n_cols_half = n_cols >> 1;
    const int n_rows_half = n_rows >> 1;
    img_dst = cv::Mat(n_rows_half, n_cols_half, CV_8U);

    // std::cout << img_src.size() << " --> " << img_dst.size() << std::endl;

    const uint8_t *p_src = img_src.ptr<uint8_t>(0);
    const uint8_t *p_src00 = p_src;
    const uint8_t *p_src01 = p_src + 1;
    const uint8_t *p_src10 = p_src + n_cols;
    const uint8_t *p_src11 = p_src + n_cols + 1;
    uint8_t *p_dst = img_dst.ptr<uint8_t>(0);
    uint8_t *p_dst_end = p_dst + n_cols_half * n_rows_half;

    while (p_dst != p_dst_end)
    {
      const uint8_t *p_src_row_end = p_src00 + n_cols;
      for (; p_src00 != p_src_row_end; ++p_dst, p_src00 += 2, p_src01 += 2, p_src10 += 2, p_src11 += 2)
      {
        *p_dst = static_cast<uint8_t>((ushort)(*p_src00 + *p_src01 + *p_src10 + *p_src11) >> 2);
        // *p_dst = *p_src00;
      }
      p_src00 += n_cols;
      p_src01 += n_cols;
      p_src10 += n_cols;
      p_src11 += n_cols;
    }
  }

  void GeneratePaddedImageByMirroring(const cv::Mat &img_src, cv::Mat &img_dst, const int pad_size)
  {
    const int n_cols = img_src.cols;
    const int n_rows = img_src.rows;
    const int pad_size_doubled = pad_size * 2;
    const int n_cols_padded = n_cols + pad_size_doubled;
    const int n_rows_padded = n_rows + pad_size_doubled;

    img_dst = cv::Mat::zeros(n_rows_padded, n_cols_padded, img_src.type());

    const uint8_t *ptr_src = img_src.ptr<uint8_t>(0);
    const uint8_t *ptr_src_end = ptr_src + n_cols * n_rows;
    uint8_t *ptr_dst = img_dst.ptr<uint8_t>(0) + pad_size * n_cols_padded + pad_size;

    // copy original
    const int size_of_row = sizeof(uint8_t) * n_cols;
    for (; ptr_src < ptr_src_end; ptr_src += n_cols, ptr_dst += n_cols_padded)
    {
      memcpy(ptr_dst, ptr_src, size_of_row);
    }

    // mirror columns
    uint8_t *ptr_dst_row = img_dst.ptr<uint8_t>(pad_size);
    uint8_t *ptr_dst_row_end = img_dst.ptr<uint8_t>(n_rows_padded - pad_size);
    for (; ptr_dst_row < ptr_dst_row_end; ptr_dst_row += n_cols_padded)
    {
      ptr_src = ptr_dst_row + pad_size;
      ptr_src_end = ptr_dst_row + pad_size_doubled;
      ptr_dst = ptr_dst_row + pad_size - 1;
      for (; ptr_src < ptr_src_end; ++ptr_src, --ptr_dst)
        *ptr_dst = *ptr_src;

      ptr_src = ptr_dst_row + n_cols;
      ptr_src_end = ptr_dst_row + n_cols + pad_size;
      ptr_dst = ptr_dst_row + n_cols_padded - 1;
      for (; ptr_src < ptr_src_end; ++ptr_src, --ptr_dst)
        *ptr_dst = *ptr_src;
    }

    // upper pad
    const int size_of_padded_row = sizeof(uint8_t) * n_cols_padded;
    ptr_src = img_dst.ptr<uint8_t>(0) + pad_size * n_cols_padded;
    ptr_src_end = img_dst.ptr<uint8_t>(0) + pad_size_doubled * n_cols_padded;
    ptr_dst = img_dst.ptr<uint8_t>(0) + (pad_size - 1) * n_cols_padded;
    for (; ptr_src < ptr_src_end; ptr_src += n_cols_padded, ptr_dst -= n_cols_padded)
    {
      memcpy(ptr_dst, ptr_src, size_of_padded_row);
    }

    // lower pad
    ptr_src = img_dst.ptr<uint8_t>(0) + (n_rows_padded - pad_size_doubled) * n_cols_padded;
    ptr_src_end = img_dst.ptr<uint8_t>(0) + (n_rows_padded - pad_size) * n_cols_padded;
    ptr_dst = img_dst.ptr<uint8_t>(0) + (n_rows_padded - 1) * n_cols_padded;
    for (; ptr_src < ptr_src_end; ptr_src += n_cols_padded, ptr_dst -= n_cols_padded)
    {
      memcpy(ptr_dst, ptr_src, size_of_padded_row);
    }
  }

  // void pad_image(struct image_t *input, struct image_t *output, uint8_t expand)
  // {
  //   image_create(output, input->w + 2 * expand, input->h + 2 * expand, input->type);

  //   uint8_t *input_buf = (uint8_t *)input->buf;
  //   uint8_t *output_buf = (uint8_t *)output->buf;

  //   // Skip first `expand` rows, iterate through next input->h rows
  //   for (uint16_t i = expand; i != (output->h - expand); i++)
  //   {

  //     // Mirror first `expand` columns
  //     for (uint8_t j = 0; j != expand; j++)
  //       output_buf[i * output->w + (expand - 1 - j)] = input_buf[(i - expand) * input->w + j];

  //     // Copy corresponding row values from input image
  //     memcpy(&output_buf[i * output->w + expand], &input_buf[(i - expand) * input->w], sizeof(uint8_t) * input->w);

  //     // Mirror last `expand` columns
  //     for (uint8_t j = 0; j != expand; j++)
  //       output_buf[i * output->w + output->w - expand + j] = output_buf[i * output->w + output->w - expand - 1 - j];
  //   }

  //   // Mirror first `expand` and last `expand` rows
  //   for (uint8_t i = 0; i != expand; i++)
  //   {
  //     memcpy(&output_buf[(expand - 1) * output->w - i * output->w], &output_buf[expand * output->w + i * output->w], sizeof(uint8_t) * output->w);
  //     memcpy(&output_buf[(output->h - expand) * output->w + i * output->w], &output_buf[(output->h - expand - 1) * output->w - i * output->w], sizeof(uint8_t) * output->w);
  //   }
  // }

  void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
                            const int max_level, const bool use_padding, const int pad_size)
  {
    const int &n_cols = img_src.cols;
    const int &n_rows = img_src.rows;

    pyramid.resize(max_level);
    img_src.copyTo(pyramid[0]);

    for (int lvl = 1; lvl < max_level; ++lvl)
    {
      const cv::Mat &img_org = pyramid[lvl - 1];
      cv::Mat &img_dst = pyramid[lvl];
      image_processing::DownsampleImage(img_org, img_dst);
    }

    if (use_padding)
    {
      if (pad_size == 0)
      {
        throw std::runtime_error("pad_size == 0");
      }
      for (int lvl = max_level - 1; lvl >= 0; --lvl)
      {
        cv::Mat img_tmp(pyramid[lvl]);
        GeneratePaddedImageByMirroring(img_tmp, pyramid[lvl], pad_size);
      }
    }
  }

  void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
                            const int max_level, const bool use_padding, const int pad_size,
                            std::vector<cv::Mat> &pyramid_du, std::vector<cv::Mat> &pyramid_dv)
  {
    const int n_cols = img_src.cols;
    const int n_rows = img_src.rows;

    pyramid.resize(max_level);
    img_src.copyTo(pyramid[0]);

    for (int lvl = 1; lvl < max_level; ++lvl)
    {
      const cv::Mat &img_org = pyramid[lvl - 1];
      cv::Mat &img_dst = pyramid[lvl];
      image_processing::DownsampleImage(img_org, img_dst);
    }

    if (use_padding)
    {
      if (pad_size == 0)
      {
        throw std::runtime_error("pad_size == 0");
      }
      for (int lvl = max_level - 1; lvl >= 0; --lvl)
      {
        cv::Mat img_tmp(pyramid[lvl]);
        GeneratePaddedImageByMirroring(img_tmp, pyramid[lvl], pad_size);
      }
    }

    pyramid_du.resize(max_level);
    pyramid_dv.resize(max_level);
    for (int lvl = 0; lvl < max_level; ++lvl)
    {
      pyramid_du[lvl] = cv::Mat(pyramid[lvl].size(), CV_16S);
      pyramid_dv[lvl] = cv::Mat(pyramid[lvl].size(), CV_16S);
    }
  }
};

namespace image_processing
{
  namespace unsafe
  {
    void InterpolateImageIntensitySameRatioHorizontalRegularPattern(
        const cv::Mat &img, const cv::Point2f &pt_center,
        float ax, int win_size,
        std::vector<float> &value_interp)
    {
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
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");
      if (!(win_size & 0x01))
        std::runtime_error("'win_size' should be an odd number!");

      const double one_minus_ax = 1.0 - ax;
      const int n_cols = img.cols;
      const int n_rows = img.rows;
      const uint8_t *ptr_img = img.ptr<uint8_t>(0);

      const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

      const int half_win_size = static_cast<int>(floor(win_size * 0.5));

      const int n_pts = win_size * win_size;
      value_interp.resize(n_pts, -1.0);

      const uint8_t *ptr_row_start = ptr_img + (pt_center0.y - half_win_size) * n_cols + pt_center0.x - half_win_size;
      const uint8_t *ptr_row_end = ptr_row_start + win_size + 2; // TODO: Patch가 화면 밖으로 나갈때!
      const uint8_t *ptr_row_final = ptr_row_start + (win_size)*n_cols;

      int counter = 0;
      std::vector<float>::iterator it_value = value_interp.begin();
      for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols)
      {
        const uint8_t *ptr = ptr_row_start;
        float I1 = *ptr;
        float I2 = *(++ptr);
        float Ia = I1 * one_minus_ax;

        ++ptr;
        for (; ptr != ptr_row_end; ++ptr)
        {
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

    void InterpolateImageIntensitySameRatioHorizontalRegularPatternArbitraryWindow(
        const cv::Mat &img, const cv::Point2f &pt_center,
        float ax, int half_left, int half_right, int half_up, int half_down,
        std::vector<float> &value_interp)
    {
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
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const double one_minus_ax = 1.0 - ax;
      const int n_cols = img.cols;
      const int n_rows = img.rows;
      const uint8_t *ptr_img = img.ptr<uint8_t>(0);

      const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

      const int win_size_horizontal = half_right + half_left + 1;
      const int win_size_vertical = half_down + half_up + 1;

      const int n_pts = win_size_horizontal * win_size_vertical;
      value_interp.resize(n_pts, -1.0);

      const uint8_t *ptr_row_start = ptr_img + (pt_center0.y - half_up) * n_cols + pt_center0.x - half_left;
      const uint8_t *ptr_row_end = ptr_row_start + win_size_horizontal + 2; // TODO: Patch가 화면 밖으로 나갈때!
      const uint8_t *ptr_row_final = ptr_row_start + (win_size_vertical)*n_cols;

      int counter = 0;
      std::vector<float>::iterator it_value = value_interp.begin();
      for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols)
      {
        const uint8_t *ptr = ptr_row_start;
        float I1 = *ptr;
        float I2 = *(++ptr);
        float Ia = I1 * one_minus_ax;

        ++ptr;
        for (; ptr != ptr_row_end; ++ptr)
        {
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

    void InterpolateImageIntensity(const cv::Mat &img, const std::vector<cv::Point2f> &pts,
                                   std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const int n_cols = img.cols;
      const int n_rows = img.rows;
      const uint8_t *ptr_img = img.ptr<uint8_t>(0);

      const int n_pts = pts.size();
      value_interp.resize(n_pts, -1.0);

      std::vector<float>::iterator it_value = value_interp.begin();
      std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
      for (; it_value != value_interp.end(); ++it_value, ++it_pt)
      {
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

    void InterpolateImageIntensitySameRatio(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax, float ay,
        std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const int n_cols = img.cols;
      const int n_rows = img.rows;
      const uint8_t *ptr_img = img.ptr<uint8_t>(0);

      const int n_pts = pts.size();
      value_interp.resize(n_pts, -1.0);

      float axay = ax * ay;

      std::vector<float>::iterator it_value = value_interp.begin();
      std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
      for (; it_value != value_interp.end(); ++it_value, ++it_pt)
      {
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

    void InterpolateImageIntensitySameRatioHorizontal(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax,
        std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const int n_cols = img.cols;
      const int n_rows = img.rows;
      const uint8_t *ptr_img = img.ptr<uint8_t>(0);

      const int n_pts = pts.size();
      value_interp.resize(n_pts, -1.0);

      std::vector<float>::iterator it_value = value_interp.begin();
      std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

      for (; it_value != value_interp.end(); ++it_value, ++it_pt)
      {
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
    };
  };
};