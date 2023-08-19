#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

// References:
//  https://github.com/JiatianWu/stereo-dso/blob/master/src/util/globalFuncs.h#L243
//  https://github.com/uzh-rpg/rpg_svo/blob/d6161063b47f36ce78252ee4c4fedf3f6d8f2898/svo/src/feature_alignment.cpp#L355
#include <iostream>
#include "opencv4/opencv2/core.hpp"
#include "eigen3/Eigen/Dense"

#include "timer.h"
/*
    I1 ax   1-ax I2
    ay   (v,u)

    1-ay
    I3           I4
*/
namespace image_processing
{
    std::string ConvertImageTypeToString(const cv::Mat &cv_image);

    float InterpolateImageIntensity(
        const cv::Mat &cv_image, const cv::Point2f &pt);

    void InterpolateImageIntensity(
        const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void InterpolateImageIntensitySameRatio(
        const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
        float ax, float ay,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void InterpolateImageIntensitySameRatioHorizontal(
        const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
        float ax,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);
    void InterpolateImageIntensitySameRatioRegularPattern(
        const cv::Mat &cv_image, const cv::Point2f &pt_center, int win_size,
        std::vector<float> &value_interp, std::vector<bool> &mask_interp);

    void DownsampleImage(const cv::Mat &img_src, cv::Mat &img_dst);
    void GeneratePaddedImageByMirroring(const cv::Mat &img_src, cv::Mat &img_dst, const int pad_size);
    void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
                              const int max_level, const bool use_padding = false, const int pad_size = 0);
    void GenerateImagePyramid(const cv::Mat &img_src, std::vector<cv::Mat> &pyramid,
                              const int max_level, const bool use_padding, const int pad_size,
                              std::vector<cv::Mat> &pyramid_du, std::vector<cv::Mat> &pyramid_dv);

    namespace unsafe
    {
        void InterpolateImageIntensity(
            const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
            std::vector<float> &value_interp);
        void InterpolateImageIntensitySameRatio(
            const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
            float ax, float ay,
            std::vector<float> &value_interp);
        void InterpolateImageIntensitySameRatioHorizontal(
            const cv::Mat &cv_image, const std::vector<cv::Point2f> &pts,
            float ax,
            std::vector<float> &value_interp);
        void InterpolateImageIntensitySameRatioHorizontalRegularPattern(
            const cv::Mat &cv_image, const cv::Point2f &pt_center,
            float ax, int win_size,
            std::vector<float> &value_interp);
        void InterpolateImageIntensitySameRatioHorizontalRegularPatternArbitraryWindow(
            const cv::Mat &cv_image, const cv::Point2f &pt_center,
            float ax, int half_left, int half_right, int half_up, int half_down,
            std::vector<float> &value_interp);
    };

    // inline void interpImageSameRatio_IntelSSE(
    //   const cv::Mat& cv_image, const std::vector<cv::Point2f>& pts,
    //   float ax, float ay,
    //   std::vector<float>& value_interp, std::vector<bool>& mask_interp);
    // inline void interpImageSameRatioHorizontal_IntelSSE(
    //   const cv::Mat& cv_image, const std::vector<cv::Point2f>& pts,
    //   float ax, float ay,
    //   std::vector<float>& value_interp, std::vector<bool>& mask_interp);
};

#endif