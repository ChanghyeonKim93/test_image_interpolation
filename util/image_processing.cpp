/*
Copyright 2023 Changhyeon Kim
*/

#include "util/image_processing.h"

namespace image_processing
{
	std::string type2str(const cv::Mat& img) 
	{
		std::string r;
		int type = img.type();

		uchar depth = type & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (type >> CV_CN_SHIFT);

		switch (depth) {
			case CV_8U:  r = "8U";   break;
			case CV_8S:  r = "8S";   break;
			case CV_16U: r = "16U";  break;
			case CV_16S: r = "16S";  break;
			case CV_32S: r = "32S";  break;
			case CV_32F: r = "32F";  break;
			case CV_64F: r = "64F";  break;
			default:     r = "User"; break;
		}

		r += "C";
		r += (chans + '0');

		return r;
	}

  float interpImage(const cv::Mat &img, const cv::Point2f &pt)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const size_t n_cols = img.cols;
    const size_t n_rows = img.rows;
    const unsigned char *ptr_img = img.ptr<unsigned char>(0);

    if (pt.x < 0 || pt.x > n_cols - 1 || pt.y < 0 || pt.y > n_rows - 1)
      return -1.0f;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = static_cast<int>(pt.x);
    int v0 = static_cast<int>(pt.y);

    float ax = uc - u0;
    float ay = vc - v0;
    float axay = ax * ay;
    int idx_I1 = v0 * n_cols + u0;

    const unsigned char *ptr = ptr_img + idx_I1;
    const float &I1 = *ptr;             // v_0n_colsu_0
    const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float value_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

    return value_interp;
  }

  void interpImage(const cv::Mat &img, const std::vector<cv::Point2f> &pts,
                   std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const size_t n_cols = img.cols;
    const size_t n_rows = img.rows;
    const unsigned char *ptr_img = img.ptr<unsigned char>(0);

    const size_t n_pts = pts.size();
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

      const unsigned char *ptr = ptr_img + idx_I1;
      const float &I1 = *ptr;             // v_0n_colsu_0
      const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
      const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
      const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

      float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

      *it_value = I_interp;
      *it_mask = true;
    }
  }

  void interpImageSameRatio(
      const cv::Mat &img, const std::vector<cv::Point2f> &pts,
      float ax, float ay,
      std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const size_t n_cols = img.cols;
    const size_t n_rows = img.rows;
    const unsigned char *ptr_img = img.ptr<unsigned char>(0);

    const size_t n_pts = pts.size();
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

      const unsigned char *ptr = ptr_img + idx_I1;
      const float &I1 = *ptr;             // v_0n_colsu_0
      const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
      const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
      const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

      float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

      *it_value = I_interp;
      *it_mask = true;
    }
  }

  void interpImageSameRatioHorizontal(
      const cv::Mat &img, const std::vector<cv::Point2f> &pts,
      float ax,
      std::vector<float> &value_interp, std::vector<bool> &mask_interp)
  {
    if (img.type() != CV_8U)
      std::runtime_error("img.type() != CV_8U");

    const size_t n_cols = img.cols;
    const size_t n_rows = img.rows;
    const unsigned char *ptr_img = img.ptr<unsigned char>(0);

    const size_t n_pts = pts.size();
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

      const unsigned char *ptr = ptr_img + idx_I1;
      const float &I1 = *ptr;     // v_0n_colsu_0
      const float &I2 = *(++ptr); // v_0n_colsu_0 + 1

      float I_interp = (I2 - I1) * ax + I1;

      *it_value = I_interp;
      *it_mask = true;
    }
  }

  void interpImageSameRatioRegularPattern(
      const cv::Mat &img, const cv::Point2f &pt_center, size_t win_size,
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

  void pyrDown(const cv::Mat &img_src, cv::Mat &img_dst)
  {
    const size_t n_cols = img_src.cols;
    const size_t n_rows = img_src.rows;

    const size_t n_cols_half = n_cols >> 1;
    const size_t n_rows_half = n_rows >> 1;
    img_dst = cv::Mat(n_rows_half, n_cols_half, CV_8U);

    // std::cout << img_src.size() << " --> " << img_dst.size() << std::endl;

    const uchar *p_src = img_src.ptr<uchar>(0);
    const uchar *p_src00 = p_src;
    const uchar *p_src01 = p_src + 1;
    const uchar *p_src10 = p_src + n_cols;
    const uchar *p_src11 = p_src + n_cols + 1;
    uchar *p_dst = img_dst.ptr<uchar>(0);
    uchar *p_dst_end = p_dst + n_cols_half * n_rows_half;

    while (p_dst != p_dst_end)
    {
      const uchar* p_src_row_end = p_src00 + n_cols;
      for (; p_src00 != p_src_row_end; ++p_dst, p_src00 += 2, p_src01 += 2, p_src10 += 2, p_src11 += 2)
      {
        *p_dst = static_cast<uchar>( (ushort)(*p_src00 + *p_src01 + *p_src10 + *p_src11) >> 2);
        // *p_dst = *p_src00;
      }
      p_src00 += n_cols;
      p_src01 += n_cols;
      p_src10 += n_cols;
      p_src11 += n_cols;
    }
  }
  void padImageByMirroring(const cv::Mat& img_src, cv::Mat& img_dst, const size_t pad_size)
  {
    const size_t n_cols = img_src.cols;
    const size_t n_rows = img_src.rows;

    const size_t n_cols_padded = n_cols + 2 * pad_size;
    const size_t n_rows_padded = n_rows + 2 * pad_size;

    std::cout << "cols, rows    : " << n_cols << ", " << n_rows << std::endl;
    std::cout << "cols, rows pad: " << n_cols_padded << ", " << n_rows_padded << std::endl;
    
  }

  void generateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
                            const size_t max_level)
  {
    const size_t n_cols = img_src.cols;
    const size_t n_rows = img_src.rows;

    pyramid.resize(max_level);
    img_src.copyTo(pyramid[0]);

    for(size_t lvl = 1; lvl < max_level; ++lvl)
    {
      const cv::Mat& img_org = pyramid[lvl-1];
      image_processing::pyrDown(img_org, pyramid[lvl]);

      size_t pad_size = 25;
      padImageByMirroring(img_org, img_dst, pad_size);

    }
  }
};

namespace image_processing
{
  namespace unsafe
  {
    void interpImageSameRatioHorizontalRegularPattern(
        const cv::Mat &img, const cv::Point2f &pt_center,
        float ax, size_t win_size,
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

      const size_t n_cols = img.cols;
      const size_t n_rows = img.rows;
      const unsigned char *ptr_img = img.ptr<unsigned char>(0);

      const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

      const size_t half_win_size = static_cast<int>(floor(win_size * 0.5));

      const size_t n_pts = win_size * win_size;
      value_interp.resize(n_pts, -1.0);

      const unsigned char *ptr_row_start = ptr_img + (pt_center0.y - half_win_size) * n_cols + pt_center0.x - half_win_size;
      const unsigned char *ptr_row_end = ptr_row_start + win_size + 2; // TODO: Patch가 화면 밖으로 나갈때!
      const unsigned char *ptr_row_final = ptr_row_start + (win_size)*n_cols;

      int counter = 0;
      std::vector<float>::iterator it_value = value_interp.begin();
      for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols)
      {
        const unsigned char *ptr = ptr_row_start;
        float I1 = *ptr;
        float I2 = *(++ptr);
        float Ia = I1 * (1.0 - ax);

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

    void interpImageSameRatioHorizontalRegularPatternArbitraryWindow(
        const cv::Mat &img, const cv::Point2f &pt_center,
        float ax, size_t half_left, size_t half_right, size_t half_up, size_t half_down,
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

      const size_t n_cols = img.cols;
      const size_t n_rows = img.rows;
      const unsigned char *ptr_img = img.ptr<unsigned char>(0);

      const cv::Point2i pt_center0(static_cast<int>(pt_center.x), static_cast<int>(pt_center.y));

      const size_t win_size_horizontal = half_right + half_left + 1;
      const size_t win_size_vertical = half_down + half_up + 1;

      const size_t n_pts = win_size_horizontal * win_size_vertical;
      value_interp.resize(n_pts, -1.0);

      const unsigned char *ptr_row_start = ptr_img + (pt_center0.y - half_up) * n_cols + pt_center0.x - half_left;
      const unsigned char *ptr_row_end = ptr_row_start + win_size_horizontal + 2; // TODO: Patch가 화면 밖으로 나갈때!
      const unsigned char *ptr_row_final = ptr_row_start + (win_size_vertical)*n_cols;

      int counter = 0;
      std::vector<float>::iterator it_value = value_interp.begin();
      for (; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols)
      {
        const unsigned char *ptr = ptr_row_start;
        float I1 = *ptr;
        float I2 = *(++ptr);
        float Ia = I1 * (1.0 - ax);

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

    void interpImage(const cv::Mat &img, const std::vector<cv::Point2f> &pts,
                     std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const size_t n_cols = img.cols;
      const size_t n_rows = img.rows;
      const unsigned char *ptr_img = img.ptr<unsigned char>(0);

      const size_t n_pts = pts.size();
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

        const unsigned char *ptr = ptr_img + idx_I1;
        const float &I1 = *ptr;             // v_0n_colsu_0
        const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
        const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
        const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

        float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

        *it_value = I_interp;
      }
    };

    void interpImageSameRatio(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax, float ay,
        std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const size_t n_cols = img.cols;
      const size_t n_rows = img.rows;
      const unsigned char *ptr_img = img.ptr<unsigned char>(0);

      const size_t n_pts = pts.size();
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

        const unsigned char *ptr = ptr_img + idx_I1;
        const float &I1 = *ptr;             // v_0n_colsu_0
        const float &I2 = *(++ptr);         // v_0n_colsu_0 + 1
        const float &I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
        const float &I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

        float I_interp = axay * (I1 - I2 - I3 + I4) + ax * (-I1 + I2) + ay * (-I1 + I3) + I1;

        *it_value = I_interp;
      }
    };

    void interpImageSameRatioHorizontal(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax,
        std::vector<float> &value_interp)
    {
      if (img.type() != CV_8U)
        std::runtime_error("img.type() != CV_8U");

      const size_t n_cols = img.cols;
      const size_t n_rows = img.rows;
      const unsigned char *ptr_img = img.ptr<unsigned char>(0);

      const size_t n_pts = pts.size();
      value_interp.resize(n_pts, -1.0);

      std::vector<float>::iterator it_value = value_interp.begin();
      std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

      for (; it_value != value_interp.end(); ++it_value, ++it_pt)
      {
        const cv::Point2f &pt = *it_pt;

        int u0 = static_cast<int>(pt.x);
        int v0 = static_cast<int>(pt.y);

        int idx_I1 = v0 * n_cols + u0;

        const unsigned char *ptr = ptr_img + idx_I1;
        const float &I1 = *ptr;     // v_0n_colsu_0
        const float &I2 = *(++ptr); // v_0n_colsu_0 + 1

        float I_interp = (I2 - I1) * ax + I1;

        *it_value = I_interp;
      }
    };
  };
};