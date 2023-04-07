#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

#include <iostream>
#include "opencv4/opencv2/core.hpp"

namespace image_processing
{
    /*
     I1 ax   1-ax I2
     ay   (v,u)

     1-ay
     I3           I4
    */
    float interpImage(
        const cv::Mat &img, const cv::Point2f &pt);

    void interpImage(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void interpImageSameRatio(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax, float ay,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void interpImageSameRatioHorizontal(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void interpImageSameRatioRegularPattern(
        const cv::Mat &img, const cv::Point2f &pt_center, size_t win_size,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);

    void interpImage_unsafe(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        std::vector<float> &value_interp);
    void interpImageSameRatio_unsafe(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax, float ay,
        std::vector<float> &value_interp);
    void interpImageSameRatioHorizontal_unsafe(
        const cv::Mat &img, const std::vector<cv::Point2f> &pts,
        float ax,
        std::vector<float> &value_interp);
    void interpImageSameRatioHorizontalRegularPattern_unsafe(
        const cv::Mat &img, const cv::Point2f &pt_center,
        float ax, size_t win_size,
        std::vector<float> &value_interp);

    // inline void interpImageSameRatio_IntelSSE(
    //   const cv::Mat& img, const std::vector<cv::Point2f>& pts,
    //   float ax, float ay,
    //   std::vector<float>& value_interp, std::vector<bool>& mask_interp);
    // inline void interpImageSameRatioHorizontal_IntelSSE(
    //   const cv::Mat& img, const std::vector<cv::Point2f>& pts,
    //   float ax, float ay,
    //   std::vector<float>& value_interp, std::vector<bool>& mask_interp);
};

#endif