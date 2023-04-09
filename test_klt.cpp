#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"

#include "timer.h"

#include "image_processing.h"
#include "klt_tracker.h"

int main()
{
  cv::Mat img = cv::imread("/home/kch/Lenna.png", cv::IMREAD_GRAYSCALE);

  const size_t n_cols = img.cols;
  const size_t n_rows = img.rows;

  timer::tic();
  size_t max_level = 4;
  std::vector<cv::Mat> img_pyr;
  image_processing::generateImagePyramid(img, img_pyr, max_level);
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