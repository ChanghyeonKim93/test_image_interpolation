#include <iostream>
#include <vector>

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"

#include "timer.h"

/*
 I1 ax   1-ax I2 
 ay   (v,u)

 1-ay
 I3           I4
*/
inline float interpImage(
  const cv::Mat& img, const cv::Point2f pt);
  
inline void interpImage(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp);
inline void interpImageSameRatio(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax, float ay,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp);
inline void interpImageSameRatioHorizontal(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp);
inline void interpImageSameRatioRegularPattern(
  const cv::Mat& img, const cv::Point2f& pt_center, size_t win_size,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp);

inline void interpImage_unsafe(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  std::vector<float>& value_interp);
inline void interpImageSameRatio_unsafe(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax, float ay,
  std::vector<float>& value_interp);
inline void interpImageSameRatioHorizontal_unsafe(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax,
  std::vector<float>& value_interp);
inline void interpImageSameRatioHorizontalRegularPattern_unsafe(
  const cv::Mat& img, const cv::Point2f& pt_center, size_t win_size,
  std::vector<float>& value_interp);

// inline void interpImageSameRatio_IntelSSE(
//   const cv::Mat& img, const std::vector<cv::Point2f>& pts,
//   float ax, float ay,
//   std::vector<float>& value_interp, std::vector<bool>& mask_interp);
// inline void interpImageSameRatioHorizontal_IntelSSE(
//   const cv::Mat& img, const std::vector<cv::Point2f>& pts,
//   float ax, float ay,
//   std::vector<float>& value_interp, std::vector<bool>& mask_interp);

int main()
{
  cv::Mat img = cv::imread("/home/kch/Lenna.png", cv::IMREAD_GRAYSCALE);

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  
  std::cout << img.cols <<", "<< img.rows << std::endl;

  // Generate patch. v-major is more faster.
  float shift = 0.35;
  int u = 270;
  int v = 250;
  int win_size = 51;
  size_t max_iter = 10000;
  int half_win_size = win_size*0.5;
  std::vector<cv::Point2f> pts_sample;
  for(float vv = v - half_win_size; vv <= v + half_win_size; vv += 1)
  {
    for(float uu = u - half_win_size; uu <= u + half_win_size; uu += 1)
    {
      pts_sample.emplace_back(uu+shift,vv);
    }
  }
  
  std::vector<float> value_interp;
  std::vector<bool> mask_interp;


  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImage(img, pts_sample, value_interp, mask_interp);
  std::cout << "interpImage: " << timer::toc(0) << " [ms]\n";
  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImage_unsafe(img, pts_sample, value_interp);
  std::cout << "interpImage_unsafe: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImageSameRatio(img, pts_sample, shift, 0.0, value_interp, mask_interp);
  std::cout << "interpImageSameRatio: " << timer::toc(0) << " [ms]\n";
  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImageSameRatio_unsafe(img, pts_sample, shift, 0.0, value_interp);
  std::cout << "interpImageSameRatio_unsafe: " << timer::toc(0) << " [ms]\n";
  
  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImageSameRatioHorizontal(img, pts_sample, shift, value_interp, mask_interp);
  std::cout << "interpImageSameRatioHorizontal: " << timer::toc(0) << " [ms]\n";
  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImageSameRatioHorizontal_unsafe(img, pts_sample, shift, value_interp);
  std::cout << "interpImageSameRatioHorizontal_unsafe: " << timer::toc(0) << " [ms]\n";

  timer::tic();
  for(int iter = 0; iter < max_iter; ++iter)
    interpImageSameRatioHorizontalRegularPattern_unsafe(img, cv::Point2f(u,v), win_size, value_interp);
  std::cout << "interpImageSameRatioHorizontalRegularPattern_unsafe: " << timer::toc(0) << " [ms]\n";

  std::cout << "length of patch : " << pts_sample.size() << std::endl;
  std::vector<unsigned char> values(value_interp.size(),255);
  for(int i = 0; i < value_interp.size(); ++i)
    values[i] = (uchar)value_interp[i];

  int sz = std::sqrt(pts_sample.size());
  cv::Mat img_interp(sz,sz,CV_8U);
  memcpy(img_interp.ptr<uchar>(0), values.data(), sizeof(unsigned char) * sz * sz);

  // Showing
  cv::Point2f tl(999,999);
  cv::Point2f br(0,0);
  
  for(const auto& pt : pts_sample)
  {
    if(tl.x > pt.x) tl.x = pt.x;
    if(tl.y > pt.y) tl.y = pt.y;
    if(br.x < pt.x) br.x = pt.x;
    if(br.y < pt.y) br.y = pt.y;
  }
  cv::Rect rect(tl,br);

  cv::rectangle(img,rect, cv::Scalar(255,255,255),1);

  cv::namedWindow("img");
  cv::imshow("img", img);

  cv::namedWindow("img_interp");  
  cv::imshow("img_interp", img_interp);

  cv::waitKey(0);

  return 1;
};

inline float interpImage(const cv::Mat& img, const cv::Point2f pt)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  if(pt.x < 0 || pt.x > n_cols-1 || pt.y < 0 || pt.y > n_rows-1) 
    return -1.0f;

  const float uc = pt.x;
  const float vc = pt.y;
  int u0 = (int)pt.x;
  int v0 = (int)pt.y;

  float ax = uc - u0;
  float ay = vc - v0;
  float axay = ax*ay;
  int idx_I1 = v0*n_cols + u0;

  const unsigned char* ptr = ptr_img + idx_I1;
  const float& I1 = *ptr;             // v_0n_colsu_0
  const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1
  const float& I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
  const float& I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

  float value_interp = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;

  return value_interp;
};

inline void interpImage(const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);
  mask_interp.resize(n_pts, false);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<bool>::iterator it_mask = mask_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for(; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;
    if(pt.x < 0 || pt.x > n_cols-1 || pt.y < 0 || pt.y > n_rows-1) 
      continue;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    float ax = uc - u0;
    float ay = vc - v0;
    float axay = ax*ay;
    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float& I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float& I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
    
    *it_value = I_interp;
    *it_mask  = true;
  }
};

inline void interpImageSameRatio(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax, float ay,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);
  mask_interp.resize(n_pts, false);

  float axay = ax*ay;

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<bool>::iterator it_mask = mask_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for(; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;
    if(pt.x < 0 || pt.x > n_cols-1 || pt.y < 0 || pt.y > n_rows-1) 
      continue;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float& I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float& I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
    
    *it_value = I_interp;
    *it_mask  = true;
  }
};

inline void interpImageSameRatioHorizontal(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);
  mask_interp.resize(n_pts, false);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<bool>::iterator it_mask = mask_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

  for(; it_value != value_interp.end(); ++it_value, ++it_mask, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;
    if(pt.x < 0 || pt.x > n_cols-1 || pt.y < 0 || pt.y > n_rows-1) 
      continue;

    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1

    float I_interp = (I2-I1)*ax + I1;
    
    *it_value = I_interp;
    *it_mask  = true;
  }
};


inline void interpImageSameRatioRegularPattern(
  const cv::Mat& img, const cv::Point2f& pt_center, size_t win_size,
  std::vector<float>& value_interp, std::vector<bool>& mask_interp)
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
 
  const cv::Point2i pt_center0((int)pt_center.x,(int)pt_center.y);
  const float ax = pt_center.x - pt_center0.x;
  const float ay = pt_center.y - pt_center0.y;
  const float axay = ax*ay;

};

inline void interpImageSameRatioHorizontalRegularPattern_unsafe(
  const cv::Mat& img, const cv::Point2f& pt_center, size_t win_size,
  std::vector<float>& value_interp)
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
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");
  if(!(win_size & 0x01)) std::runtime_error("'win_size' should be an odd number!");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const cv::Point2i pt_center0((int)pt_center.x,(int)pt_center.y);
  const float ax = pt_center.x - pt_center0.x;
 
  const size_t half_win_size = (int)floor(win_size*0.5);

  const size_t n_pts = win_size * win_size;
  value_interp.resize(n_pts, -1.0);

  const unsigned char* ptr_row_start = ptr_img + (pt_center0.y - half_win_size)*n_cols + pt_center0.x - half_win_size - 1;
  const unsigned char* ptr_row_end   = ptr_row_start + win_size + 1; // TODO: Patch가 화면 밖으로 나갈때!
  const unsigned char* ptr_row_final = ptr_row_start + (win_size + 1) * n_cols;

  int counter = 0;
  std::vector<float>::iterator it_value = value_interp.begin();
  for(; ptr_row_start != ptr_row_final; ptr_row_start += n_cols, ptr_row_end += n_cols)
  {
    const unsigned char* ptr = ptr_row_start;
    float I1 = *ptr;
    float I2 = *(++ptr);
    ++ptr;

    float Ia = I1*(1.0-ax);
    for(; ptr != ptr_row_end;)
    {
      float Ib = I2*ax;
      float I_interp = Ia + Ib; 
      
      *(it_value) = I_interp;      

      Ia = I2 - Ib;
      I2 = *(++ptr);
      ++it_value;
      ++counter;
    }

  }
  std::cout << "counter: " << counter <<", npts: " << n_pts << std::endl;
};

inline void interpImage_unsafe(const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  std::vector<float>& value_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for(; it_value != value_interp.end(); ++it_value, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    float ax = uc - u0;
    float ay = vc - v0;
    float axay = ax*ay;
    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float& I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float& I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
    
    *it_value = I_interp;
  }
};

inline void interpImageSameRatio_unsafe(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax, float ay,
  std::vector<float>& value_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  float axay = ax*ay;

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();
  for(; it_value != value_interp.end(); ++it_value, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;

    const float uc = pt.x;
    const float vc = pt.y;
    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1
    const float& I4 = *(ptr += n_cols); // v_0n_colsu_0 + 1 + n_cols
    const float& I3 = *(--ptr);         // v_0n_colsu_0 + n_cols

    float I_interp = axay*(I1 - I2 - I3 + I4) + ax*(-I1 + I2) + ay*(-I1 + I3) + I1;
    
    *it_value = I_interp;
  }
};

inline void interpImageSameRatioHorizontal_unsafe(
  const cv::Mat& img, const std::vector<cv::Point2f>& pts,
  float ax,
  std::vector<float>& value_interp)
{
  if(img.type() != CV_8U) std::runtime_error("img.type() != CV_8U");

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;
  const unsigned char* ptr_img = img.ptr<unsigned char>(0);

  const size_t n_pts = pts.size();
  value_interp.resize(n_pts, -1.0);

  std::vector<float>::iterator it_value = value_interp.begin();
  std::vector<cv::Point2f>::const_iterator it_pt = pts.begin();

  for(; it_value != value_interp.end(); ++it_value, ++it_pt)
  {
    const cv::Point2f& pt = *it_pt;

    int u0 = (int)pt.x;
    int v0 = (int)pt.y;

    int idx_I1 = v0*n_cols + u0;

    const unsigned char* ptr = ptr_img + idx_I1;
    const float& I1 = *ptr;             // v_0n_colsu_0
    const float& I2 = *(++ptr);         // v_0n_colsu_0 + 1

    float I_interp = (I2-I1)*ax + I1;
    
    *it_value = I_interp;
  }
};

