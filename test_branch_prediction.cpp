#include <iostream>
#include <exception>
#include <string>
#include <random>

#include "opencv4/opencv2/core.hpp"

#include "util/timer.h"

// valgrind --tool=cachegrind --branch-sim=yes ./test_branch_prediction
int main()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0,255);

	size_t max_iterations = 100;
	try{
		size_t n_data = 1e6;
		std::vector<size_t> values(n_data, 0);
		for(auto &it : values)
			it = dist(gen);

		uint8_t threshold = 127;

		timer::tic();
		std::sort(values.begin(), values.end());
		std::vector<size_t> res1;
		for(size_t iter = 0; iter < max_iterations; ++iter){
			size_t sum = 0;
			for( const auto & it: values) {
				if(it > threshold)
					sum += it;
			}
			res1.push_back(sum);
		}
		timer::toc(1);
	}
	catch (std::runtime_error e)
	{
		std::cout << " === Runtime error: " << e.what();
	}

  return 1;
}