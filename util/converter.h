#ifndef _CONVERTER_H_
#define _CONVERTER_H_

#include <iostream>
#include <opencv4/opencv2/core.hpp>

namespace converter {
  void convertCvKeyPointToCvPoint(
    const std::vector<cv::KeyPoint>& kpts,
    std::vector<cv::Point2f>& pts);
};

#endif