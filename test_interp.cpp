#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "util/timer.h"
#include "util/image_processing.h"

int main()
{
//  cv::Mat img = cv::imread("Lenna.png", cv::IMREAD_GRAYSCALE);
 cv::Mat img = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);

 const size_t n_cols = img.cols;
 const size_t n_rows = img.rows;

 std::cout << img.cols << ", " << img.rows << std::endl;

 // Generate patch. v-major is more faster.
 float shift = 0.44;
 int u = 350;
 int v = 250;
 int win_size = 51;
 size_t max_iter = 10000;
 int half_win_size = win_size * 0.5;
 std::vector<cv::Point2f> pts_sample;
 for (float vv = v - half_win_size; vv <= v + half_win_size; vv += 1)
 {
   for (float uu = u - half_win_size; uu <= u + half_win_size; uu += 1)
   {
     pts_sample.emplace_back(uu + shift, vv);
   }
 }

 std::vector<std::vector<float>> value_interp(30);
 std::vector<float> value_interp2;
 std::vector<bool> mask_interp;

 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::interpImage(img, pts_sample, value_interp[0], mask_interp);
 std::cout << "interpImage: " << timer::toc(0) << " [ms]\n";
 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::unsafe::interpImage(img, pts_sample, value_interp[1]);
 std::cout << "interpImage_unsafe: " << timer::toc(0) << " [ms]\n";

 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::interpImageSameRatio(img, pts_sample, shift, 0.0, value_interp[2], mask_interp);
 std::cout << "interpImageSameRatio: " << timer::toc(0) << " [ms]\n";
 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::unsafe::interpImageSameRatio(img, pts_sample, shift, 0.0, value_interp[3]);
 std::cout << "interpImageSameRatio_unsafe: " << timer::toc(0) << " [ms]\n";

 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::interpImageSameRatioHorizontal(img, pts_sample, shift, value_interp[4], mask_interp);
 std::cout << "interpImageSameRatioHorizontal: " << timer::toc(0) << " [ms]\n";
 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::unsafe::interpImageSameRatioHorizontal(img, pts_sample, shift, value_interp[5]);
 std::cout << "interpImageSameRatioHorizontal_unsafe: " << timer::toc(0) << " [ms]\n";

 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::unsafe::interpImageSameRatioHorizontalRegularPattern(img, cv::Point2f(u + shift, v), shift, win_size, value_interp[6]);
 std::cout << "interpImageSameRatioHorizontalRegularPattern_unsafe: " << timer::toc(0) << " [ms]\n";

 timer::tic();
 for (int iter = 0; iter < max_iter; ++iter)
   image_processing::unsafe::interpImageSameRatioHorizontalRegularPatternArbitraryWindow(img, cv::Point2f(u + shift, v), shift, 25, 25, 25, 25, value_interp[7]);
 std::cout << "interpImageSameRatioHorizontalRegularPatternArbitraryWindow_unsafe: " << timer::toc(0) << " [ms]\n";

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
 cv::Mat img_interp(sz, sz, CV_8U);
 memcpy(img_interp.ptr<uchar>(0), values.data(), sizeof(unsigned char) * sz * sz);

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

 cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 1);

 cv::namedWindow("img");
 cv::imshow("img", img);

 cv::namedWindow("img_interp");
 cv::imshow("img_interp", img_interp);

 cv::waitKey(0);

 return 1;
};
