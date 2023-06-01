#include "util/feature_extractor.h"

FeatureExtractor::FeatureExtractor(
	const size_t max_num_features, const float threshold_fast, const float scale_factor, const int num_level)
: threshold_fast_(threshold_fast), scale_factor_(scale_factor), num_level_(num_level)
{
	this->extractor_orb_ = cv::ORB::create();

	std::cout << " ORB is created.\n";

	extractor_orb_->setMaxFeatures(10000);
	extractor_orb_->setScaleFactor(this->scale_factor_);
	extractor_orb_->setNLevels(this->num_level_);
	extractor_orb_->setEdgeThreshold(31);
	extractor_orb_->setFirstLevel(0);
	extractor_orb_->setWTA_K(2);
	extractor_orb_->setScoreType(cv::ORB::HARRIS_SCORE);
	extractor_orb_->setPatchSize(31);
	extractor_orb_->setFastThreshold(this->threshold_fast_);

	std::cout << " - FEATURE_EXTRACTOR is constructed.\n";
};

FeatureExtractor::~FeatureExtractor() 
{
	std::cout << " - FEATURE_EXTRACTOR is deleted.\n";
}

void FeatureExtractor::setThresholdFast(const float threshold_fast){
	threshold_fast_ = threshold_fast;
	extractor_orb_->setFastThreshold(threshold_fast_);
}

void FeatureExtractor::extractORB(const cv::Mat& img, PixelVec& pts_extracted) {
	// INPUT IMAGE MUST BE CV_8UC1 image.
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	cv::Mat desc_image;
	std::vector<cv::KeyPoint> kpts_extracted;
	extractor_orb_->detect(img_in, kpts_extracted);

	const size_t n_pts = kpts_extracted.size();
	if(n_pts == 0) return;

	pts_extracted.resize(n_pts);
	for(size_t index = 0; index < n_pts; ++index) {
		pts_extracted[index] = kpts_extracted[index].pt;
	}
}
void FeatureExtractor::extractORBwithBinning(const cv::Mat& img, 
	const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
	PixelVec& pts_extracted)
{
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	if(n_maximum_feature_per_bin < 1)
		throw std::runtime_error("In FeatureExtractor::extractAndComputORBwithBinning, n_maximum_feature_per_bin < 1");

	const int n_cols = img_in.cols;
	const int n_rows = img_in.rows;
	const double inv_n_cols = 1.0 / static_cast<double>(n_cols);
	const double inv_n_rows = 1.0 / static_cast<double>(n_rows);

	const double bin_size_column = static_cast<double>(n_cols) / static_cast<double>(n_bins_u);
	const double bin_size_row = static_cast<double>(n_rows) / static_cast<double>(n_bins_v);
	const double inverse_bin_size_column = 1.0 / bin_size_column;
	const double inverse_bin_size_row = 1.0 / bin_size_row;

	const int n_total_bins = n_bins_u * n_bins_v;
	
	// extract all features
	std::vector<cv::KeyPoint> kpts_all;
	extractor_orb_->detect(img_in, kpts_all);

	const size_t n_pts = kpts_all.size();
	if(n_pts == 0) return;

	// Index Bucketing
	struct IndexResponse {
		size_t index;
		float response;
	};
	struct FeatureBin {
		std::vector<IndexResponse> index_response_vector;
		bool is_preoccupied{false};
	};

	std::vector<FeatureBin> feature_bins(n_total_bins);
	for(size_t index_feature = 0; index_feature < n_pts; ++index_feature) {
		const cv::KeyPoint& kpt = kpts_all[index_feature];

		const int bin_column = static_cast<int>(kpt.pt.x * inverse_bin_size_column);
		const int bin_row = static_cast<int>(kpt.pt.y * inverse_bin_size_row);

		if(bin_column < 0 || bin_column >= n_bins_u) continue;
		if(bin_row < 0 || bin_row >= n_bins_v) continue;

		const size_t index_bin = bin_column + n_bins_u * bin_row;
		feature_bins[index_bin].index_response_vector.push_back({index_feature, kpt.response});
	}

	// sort and remain fixed number of features per bin
	auto compare_functor = [](const IndexResponse& a, const IndexResponse& b) {
		return a.response < b.response;
	};
	
	for(size_t index_bin = 0; index_bin < n_total_bins; ++index_bin) {
		std::vector<IndexResponse>& indexes_and_responses = feature_bins[index_bin].index_response_vector;

		if(indexes_and_responses.size() == 0) continue;

		if(indexes_and_responses.size() > 1) {
			std::sort(indexes_and_responses.begin(), indexes_and_responses.end(), compare_functor);
			if(indexes_and_responses.size() > n_maximum_feature_per_bin)
				indexes_and_responses.resize(n_maximum_feature_per_bin);
		}
	}

	pts_extracted.resize(0);
	pts_extracted.reserve(n_total_bins);
	for(const FeatureBin& feature_bin : feature_bins) {
		const std::vector<IndexResponse>& indexes_and_responses = feature_bin.index_response_vector;
		if(indexes_and_responses.empty()) continue;

		for(const IndexResponse& index_and_response : indexes_and_responses) {
			const cv::KeyPoint& kpt = kpts_all[index_and_response.index];
			pts_extracted.push_back(kpt.pt);
		}
	}
}

void FeatureExtractor::extractORBfromEmptyBinOnly(const cv::Mat& img,
	const std::vector<cv::Point2f>& pts_exist, 
	const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
	PixelVec& pts_extracted)
{
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	if(n_maximum_feature_per_bin < 1)
		throw std::runtime_error("In FeatureExtractor::extractORBfromEmptyBinOnly, n_maximum_feature_per_bin < 1");

	const int n_cols = img_in.cols;
	const int n_rows = img_in.rows;
	const double inv_n_cols = 1.0 / static_cast<double>(n_cols);
	const double inv_n_rows = 1.0 / static_cast<double>(n_rows);

	const double bin_size_column = static_cast<double>(n_cols) / static_cast<double>(n_bins_u);
	const double bin_size_row = static_cast<double>(n_rows) / static_cast<double>(n_bins_v);
	const double inverse_bin_size_column = 1.0 / bin_size_column;
	const double inverse_bin_size_row = 1.0 / bin_size_row;

	const int n_total_bins = n_bins_u * n_bins_v;
	
	// extract all features
	std::vector<cv::KeyPoint> kpts_all;
	extractor_orb_->detect(img_in, kpts_all);

	const size_t n_pts = kpts_all.size();
	if(n_pts == 0) return;

	// Index Bucketing
	struct IndexResponse {
		size_t index{0};
		float response{0};
	};
	struct FeatureBin {
		std::vector<IndexResponse> index_response_vector;
		bool is_preoccupied{false};
	};
	std::vector<FeatureBin> feature_bins(n_total_bins);

	// Update bin occupancy
	size_t count_preoccupied = 0;
	if(pts_exist.empty()) {
		for(size_t i = 0; i < pts_exist.size(); ++i) {
			const cv::Point2f& pt = pts_exist[i];
			const int bin_column = static_cast<int>(pt.x * inverse_bin_size_column);
			const int bin_row = static_cast<int>(pt.y * inverse_bin_size_row);

			if(bin_column < 0 || bin_column >= n_bins_u) continue;
			if(bin_row < 0 || bin_row >= n_bins_v) continue;

			const size_t index_bin = bin_column + n_bins_u * bin_row;
			feature_bins[index_bin].is_preoccupied = true;
			++count_preoccupied;
		}
	}

	for(size_t index_feature = 0; index_feature < n_pts; ++index_feature) {
		const cv::KeyPoint& kpt = kpts_all[index_feature];

		const int bin_column = static_cast<int>(kpt.pt.x * inverse_bin_size_column);
		const int bin_row = static_cast<int>(kpt.pt.y * inverse_bin_size_row);

		if(bin_column < 0 || bin_column >= n_bins_u) continue;
		if(bin_row < 0 || bin_row >= n_bins_v) continue;

		const size_t index_bin = bin_column + n_bins_u * bin_row;
		if(feature_bins[index_bin].is_preoccupied) continue;
		
		feature_bins[index_bin].index_response_vector.push_back({index_feature, kpt.response});
	}

	// sort and remain fixed number of features per bin
	auto compare_functor = [](const IndexResponse& a, const IndexResponse& b) {
		return a.response < b.response;
	};
	
	for(size_t index_bin = 0; index_bin < n_total_bins; ++index_bin) {
		FeatureBin& feature_bin = feature_bins[index_bin];

		if(feature_bin.is_preoccupied || feature_bin.index_response_vector.size() == 0) continue;

		if(feature_bin.index_response_vector.size() > 1) {
			std::sort(feature_bin.index_response_vector.begin(), feature_bin.index_response_vector.end(), compare_functor);
			if(feature_bin.index_response_vector.size() > n_maximum_feature_per_bin)
				feature_bin.index_response_vector.resize(n_maximum_feature_per_bin);
		}
	}

	pts_extracted.resize(0);
	pts_extracted.reserve(n_total_bins);
	for(const FeatureBin& feature_bin : feature_bins) {
		if(feature_bin.index_response_vector.empty()) continue;

		for(const IndexResponse& index_and_response : feature_bin.index_response_vector) {
			const cv::KeyPoint& kpt = kpts_all[index_and_response.index];
			pts_extracted.push_back(kpt.pt);
		}
	}
}
	
void FeatureExtractor::extractAndComputeORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted) {
	// INPUT IMAGE MUST BE CV_8UC1 image.
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	cv::Mat desc_image;
	extractor_orb_->detectAndCompute(img_in, cv::noArray(), kpts_extracted, desc_image);

	const size_t n_pts = kpts_extracted.size();
	if(n_pts == 0) return;

	desc_extracted.resize(n_pts);
	for(size_t index = 0; index < n_pts; ++index) {
		desc_extracted[index] = desc_image.row(index);
	}	
}

void FeatureExtractor::extractAndComputORBwithBinning(
	const cv::Mat& img, const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin,
	std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted) {
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	if(n_maximum_feature_per_bin < 1)
		throw std::runtime_error("In FeatureExtractor::extractAndComputORBwithBinning, n_maximum_feature_per_bin < 1");

	const int n_cols = img_in.cols;
	const int n_rows = img_in.rows;
	const double inv_n_cols = 1.0 / static_cast<double>(n_cols);
	const double inv_n_rows = 1.0 / static_cast<double>(n_rows);

	const double bin_size_column = static_cast<double>(n_cols) / static_cast<double>(n_bins_u);
	const double bin_size_row = static_cast<double>(n_rows) / static_cast<double>(n_bins_v);
	const double inverse_bin_size_column = 1.0 / bin_size_column;
	const double inverse_bin_size_row = 1.0 / bin_size_row;

	const int n_total_bins = n_bins_u * n_bins_v;
	
	// extract all features
	cv::Mat desc_all_cv;
	std::vector<cv::KeyPoint> kpts_all;
	std::vector<cv::Mat> desc_all;
	extractor_orb_->detectAndCompute(img_in, cv::noArray(), kpts_all, desc_all_cv);

	const size_t n_pts = kpts_all.size();
	if(n_pts == 0) return;

	desc_all.resize(n_pts);
	for(size_t i = 0; i < n_pts; ++i) {
		desc_all[i] = desc_all_cv.row(i);
	}

	// Index Bucketing
	struct IndexResponse {
		size_t index{0};
		float response{0};
	};
	struct FeatureBin {
		std::vector<IndexResponse> index_response_vector;
		bool is_preoccupied{false};
	};
	std::vector<FeatureBin> feature_bins(n_total_bins);
	for(size_t index_feature = 0; index_feature < n_pts; ++index_feature) {
		const cv::KeyPoint& kpt = kpts_all[index_feature];

		const int bin_column = static_cast<int>(kpt.pt.x * inverse_bin_size_column);
		const int bin_row = static_cast<int>(kpt.pt.y * inverse_bin_size_row);

		if(bin_column < 0 || bin_column >= n_bins_u) continue;
		if(bin_row < 0 || bin_row >= n_bins_v) continue;

		const size_t index_bin = bin_column + n_bins_u * bin_row;
		feature_bins[index_bin].index_response_vector.push_back({index_feature, kpt.response});
	}

	// sort and remain fixed number of features per bin
	auto compare_functor = [](const IndexResponse& a, const IndexResponse& b) {
		return a.response < b.response;
	};
	
	for(size_t index_bin = 0; index_bin < n_total_bins; ++index_bin) {
		std::vector<IndexResponse>& indexes_and_responses = feature_bins[index_bin].index_response_vector;

		if(indexes_and_responses.size() == 0) continue;

		if(indexes_and_responses.size() > 1) {
			std::sort(indexes_and_responses.begin(), indexes_and_responses.end(), compare_functor);
			if(indexes_and_responses.size() > n_maximum_feature_per_bin)
				indexes_and_responses.resize(n_maximum_feature_per_bin);
		}
	}

	kpts_extracted.resize(0);
	desc_extracted.resize(0);
	kpts_extracted.reserve(n_total_bins);
	desc_extracted.reserve(n_total_bins);
	for(const FeatureBin& feature_bin : feature_bins) {
		const std::vector<IndexResponse>& indexes_and_responses = feature_bin.index_response_vector;
		if(indexes_and_responses.empty()) continue;

		for(const IndexResponse& index_and_response : indexes_and_responses) {
			const cv::KeyPoint& kpt = kpts_all[index_and_response.index];
			const cv::Mat& desc = desc_all[index_and_response.index];
			kpts_extracted.push_back(kpt);
			desc_extracted.push_back(desc);
		}
	}
}


void FeatureExtractor::extractAndComputORBfromEmptyBinOnly(
	const cv::Mat& img,
	const std::vector<cv::Point2f>& pts_exist, 
	const int n_bins_u, const int n_bins_v, const int n_maximum_feature_per_bin, 
	std::vector<cv::KeyPoint>& kpts_extracted, std::vector<cv::Mat>& desc_extracted)
{
	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else img_in = img;

	if(n_maximum_feature_per_bin < 1)
		throw std::runtime_error("In FeatureExtractor::extractAndComputORBfromEmptyBinOnly, n_maximum_feature_per_bin < 1");

	const int n_cols = img_in.cols;
	const int n_rows = img_in.rows;
	const double inv_n_cols = 1.0 / static_cast<double>(n_cols);
	const double inv_n_rows = 1.0 / static_cast<double>(n_rows);

	const double bin_size_column = static_cast<double>(n_cols) / static_cast<double>(n_bins_u);
	const double bin_size_row = static_cast<double>(n_rows) / static_cast<double>(n_bins_v);
	const double inverse_bin_size_column = 1.0 / bin_size_column;
	const double inverse_bin_size_row = 1.0 / bin_size_row;

	const int n_total_bins = n_bins_u * n_bins_v;
	
	// extract all features
	cv::Mat desc_all_cv;
	std::vector<cv::KeyPoint> kpts_all;
	std::vector<cv::Mat> desc_all;
	extractor_orb_->detectAndCompute(img_in, cv::noArray(), kpts_all, desc_all_cv);

	const size_t n_pts = kpts_all.size();
	if(n_pts == 0) return;

	desc_all.resize(n_pts);
	for(size_t i = 0; i < n_pts; ++i) {
		desc_all[i] = desc_all_cv.row(i);
	}

	// Index Bucketing
	struct IndexResponse {
		size_t index{0};
		float response{0};
	};
	struct FeatureBin {
		std::vector<IndexResponse> index_response_vector;
		bool is_preoccupied{false};
	};
	
	std::vector<FeatureBin> feature_bins(n_total_bins);

	// Update bin occupancy
	size_t count_preoccupied = 0;
	if(pts_exist.empty()) {
		for(size_t i = 0; i < pts_exist.size(); ++i) {
			const cv::Point2f& pt = pts_exist[i];
			const int bin_column = static_cast<int>(pt.x * inverse_bin_size_column);
			const int bin_row = static_cast<int>(pt.y * inverse_bin_size_row);

			if(bin_column < 0 || bin_column >= n_bins_u) continue;
			if(bin_row < 0 || bin_row >= n_bins_v) continue;

			const size_t index_bin = bin_column + n_bins_u * bin_row;
			feature_bins[index_bin].is_preoccupied = true;
			++count_preoccupied;
		}
	}

	for(size_t index_feature = 0; index_feature < n_pts; ++index_feature) {
		const cv::KeyPoint& kpt = kpts_all[index_feature];

		const int bin_column = static_cast<int>(kpt.pt.x * inverse_bin_size_column);
		const int bin_row = static_cast<int>(kpt.pt.y * inverse_bin_size_row);

		if(bin_column < 0 || bin_column >= n_bins_u) continue;
		if(bin_row < 0 || bin_row >= n_bins_v) continue;

		const size_t index_bin = bin_column + n_bins_u * bin_row;
		if(feature_bins[index_bin].is_preoccupied) continue;

		feature_bins[index_bin].index_response_vector.push_back({index_feature, kpt.response});
	}

	// sort and remain fixed number of features per bin
	auto compare_functor = [](const IndexResponse& a, const IndexResponse& b) {
		return a.response < b.response;
	};
	
	for(size_t index_bin = 0; index_bin < n_total_bins; ++index_bin) {
		if(feature_bins[index_bin].is_preoccupied) continue;

		std::vector<IndexResponse>& indexes_and_responses = feature_bins[index_bin].index_response_vector;

		if(indexes_and_responses.size() == 0) continue;

		if(indexes_and_responses.size() > 1) {
			std::sort(indexes_and_responses.begin(), indexes_and_responses.end(), compare_functor);
			if(indexes_and_responses.size() > n_maximum_feature_per_bin)
				indexes_and_responses.resize(n_maximum_feature_per_bin);
		}
	}

	kpts_extracted.resize(0);
	desc_extracted.resize(0);
	kpts_extracted.reserve(n_total_bins);
	desc_extracted.reserve(n_total_bins);
	for(const FeatureBin& feature_bin : feature_bins) {
		const std::vector<IndexResponse>& indexes_and_responses = feature_bin.index_response_vector;
		if(indexes_and_responses.empty()) continue;

		for(const IndexResponse& index_and_response : indexes_and_responses) {
			const cv::KeyPoint& kpt = kpts_all[index_and_response.index];
			const cv::Mat& desc = desc_all[index_and_response.index];
			kpts_extracted.push_back(kpt);
			desc_extracted.push_back(desc);
		}
	}
}

void FeatureExtractor::calculateHarrisScore(
	const cv::Mat& img,
	const std::vector<cv::Point2f>& pts, const size_t window_size, 
	std::vector<float>& scores)
{
	const double kappa = 0.01;

	const int n_elem = window_size*window_size;

	const size_t n_cols = img.cols;
	const size_t n_rows = img.rows;
	cv::Mat du;
	cv::Mat dv;
	cv::Sobel(img,du,CV_16S,1,0,3);
	cv::Sobel(img,dv,CV_16S,0,1,3);

	const size_t half_window_size = window_size/2;
	const size_t n_pts = pts.size();
	scores.resize(n_pts);
	for(size_t index = 0; index < n_pts; ++index) {
		const cv::Point2f& pt = pts[index];

		const int u = std::floor(pt.x);
		const int v = std::floor(pt.y);
		if(u < half_window_size || u >= n_cols - half_window_size ||
		 	 v < half_window_size || v >= n_rows - half_window_size)
		{
			scores[index] = 0;
			continue;
		}
		cv::Rect roi = cv::Rect(cv::Point(u-half_window_size,v-half_window_size),cv::Point(u+half_window_size,v+half_window_size));
		cv::Mat du_pattern = du(roi);
		cv::Mat dv_pattern = dv(roi);

		du_pattern /= 2048;
		dv_pattern /= 2048;

		double r11 = 0, r12 = 0, r22 = 0;
		short* ptr_du = du_pattern.ptr<short>(0);
		short* ptr_dv = dv_pattern.ptr<short>(0);
		const short* ptr_du_end = ptr_du + n_elem;
		const short* ptr_dv_end = ptr_dv + n_elem;

		for(; ptr_du != ptr_du_end; ++ptr_du, ++ptr_dv){
			r11 += (*ptr_du)*(*ptr_du);
			r12 += (*ptr_du)*(*ptr_dv);
			r22 += (*ptr_dv)*(*ptr_dv);
		}
		r11 /= n_elem;
		r12 /= n_elem;
		r22 /= n_elem;
		double det = r11*r22 - r12*r12;
		double tr = r11 + r22;

		scores[index] = det - kappa*tr;
				
	}
}