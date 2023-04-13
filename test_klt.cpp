#define _ENABLE_EXTENDED_ALIGNED_STORAGE

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "util/timer.h"

#include "util/image_processing.h"
#include "util/klt_tracker.h"

#include "util/image_float.h"

int main()
{
  cv::Mat img = cv::imread("C:\\Users\\rlack\\Source\\Repos\\test_img_interp\\test_img_interp\\test_image_interpolation\\Lenna.png", cv::IMREAD_GRAYSCALE);

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;

  Image<float> image(n_cols, n_rows);
  image.fillZero();

  Image<float> image2;
  image2 = image;

  std::cout << image.data() << std::endl;
  std::cout << image2.data() << std::endl;
  

  /*
    pyrDown: sampling vs. averaging
    klt: forward vs. inverse
    klt: dense patch vs. sparse patch
  */
  std::shared_ptr<KLTTracker> klt_tracker = std::make_shared<KLTTracker>();
  std::vector<cv::Point2f> pts0;
  std::vector<cv::Point2f> pts_track;
  pts0.push_back({20, 20});
  klt_tracker->trackForwardAdditive(img, img, pts0, pts_track, 4, 25, 30, false, false);
  // klt_tracker->trackForwardAdditive(img, img, pts0, pts_track, 4, 25, 30, true, false);

  timer::tic();
  size_t max_level = 5;
  std::vector<cv::Mat> img_pyr;
  image_processing::generateImagePyramid(img, img_pyr, max_level, true, 12);
  timer::toc(1);
  for (int lvl = 0; lvl < max_level; ++lvl)
  {
    std::string str_winname = "img" + std::to_string(lvl);
    cv::namedWindow(str_winname);
    cv::imshow(str_winname, img_pyr[lvl]);
  }
  cv::waitKey(0);

  return 1;
}