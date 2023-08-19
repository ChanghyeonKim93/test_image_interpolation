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
	try{
		// cv::Mat img = cv::imread("C:\\Users\\rlack\\Source\\Repos\\test_img_interp\\test_img_interp\\test_image_interpolation\\Lenna.png", cv::IMREAD_GRAYSCALE);
 		cv::Mat img = cv::imread("/home/kch/Documents/test_image_interpolation/Lenna.png", cv::IMREAD_GRAYSCALE);

		const size_t n_cols = 8;
		const size_t n_rows = 8;

		Image<float> image(n_cols, n_rows);
		std::cout << image << std::endl;

		image.fillOne();
		Image<float> image2;
		image2 = image;
		std::cout << image2 << std::endl;

		image2.fill(2.0f);
		std::cout << image2 << std::endl;

		image2 = (image2 + image);
		std::cout << image2 << std::endl;

		image2 += image2;
		std::cout << image2 << std::endl;

		ObjectPool<Node> node_pool(1000);


		/*
			pyrDown: sampling vs. averaging
			klt: forward vs. inverse
			klt: dense patch vs. sparse patch
		*/
		std::shared_ptr<KLTTracker> klt_tracker = std::make_shared<KLTTracker>();
		std::vector<cv::Point2f> pts0;
		std::vector<cv::Point2f> pts_track;
		pts0.push_back({ 20, 20 });
		klt_tracker->trackForwardAdditive(img, img, pts0, pts_track, 4, 25, 30, false, false);
		// klt_tracker->trackForwardAdditive(img, img, pts0, pts_track, 4, 25, 30, true, false);

		timer::tic();
		size_t max_level = 5;
		std::vector<cv::Mat> img_pyr;
		image_processing::GenerateImagePyramid(img, img_pyr, max_level, true, 12);
		timer::toc(1);
		for (int lvl = 0; lvl < max_level; ++lvl)
		{
			std::string str_winname = "img" + std::to_string(lvl);
			cv::namedWindow(str_winname);
			cv::imshow(str_winname, img_pyr[lvl]);
		}
		cv::waitKey(0);

	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}