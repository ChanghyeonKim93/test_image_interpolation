#include <iostream>
#include <exception>
#include <string>
#include <random>

#include "opencv4/opencv2/core.hpp"

#include "util/timer.h"

// valgrind --tool=cachegrind ./test_cache_miss
int main()
{
	try{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0,255);	

		size_t n_cols = 1920;
		size_t n_rows = 1080;
		cv::Mat img_uint8 = cv::Mat::zeros(cv::Size(n_cols,n_rows), CV_8U);

		unsigned char* ptr = img_uint8.data;
		unsigned char* ptr_end = ptr + img_uint8.step * img_uint8.rows;
		for (; ptr < ptr_end; ++ptr) {
			*ptr = (unsigned char) dist(gen);
		}

		// row-major test
		size_t max_iterations = 100;
		
		timer::tic();
		std::vector<size_t> res1;
		for(size_t iter = 0; iter < max_iterations; ++iter) {
			size_t sum = 0;
			for(size_t u = 0; u < img_uint8.cols; ++u) {
				unsigned char* ptr = img_uint8.data + u;
				const unsigned char* ptr_end = ptr + u + (img_uint8.rows - 1) * img_uint8.cols;
				for(; ptr <= ptr_end; ptr += img_uint8.cols)
					sum += *ptr;

				res1.push_back(sum);
			}
		}
		timer::toc(1);
	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}