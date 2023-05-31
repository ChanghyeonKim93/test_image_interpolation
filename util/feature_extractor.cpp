#include "util/feature_extractor.h"

FeatureExtractor::FeatureExtractor() 
: params_orb_() 
{
	weight_bin_  = nullptr;
	flag_debug_  = false;
	flag_nonmax_ = true;

	n_bins_u_   = 0;
	n_bins_v_   = 0;

	THRES_FAST_ = 0;
	r_          = 0;

	NUM_NONMAX_ = 0;

	std::cout << " - FEATURE_EXTRACTOR is constructed.\n";
};

FeatureExtractor::~FeatureExtractor() 
{
	std::cout << " - FEATURE_EXTRACTOR is deleted.\n";
};

void FeatureExtractor::initParams(int n_cols, int n_rows, int n_bins_u, int n_bins_v, int THRES_FAST, int radius) 
{
	std::cout << " - FEATURE_EXTRACTOR - 'initParams'\n";

	this->extractor_orb_ = cv::ORB::create();

	std::cout << " ORB is created.\n";

	this->flag_debug_  = false;
	this->flag_nonmax_ = true;

	this->NUM_NONMAX_  = 5;

	this->n_bins_u_    = n_bins_u;
	this->n_bins_v_    = n_bins_v;

	this->THRES_FAST_  = THRES_FAST;
	this->r_           = radius;

	this->params_orb_.FastThreshold = this->THRES_FAST_;
	this->params_orb_.n_bins_u      = this->n_bins_u_;
	this->params_orb_.n_bins_v      = this->n_bins_v_;

	extractor_orb_->setMaxFeatures(10000);
	extractor_orb_->setScaleFactor(1.2);
	extractor_orb_->setNLevels(8);
	extractor_orb_->setEdgeThreshold(31);
	extractor_orb_->setFirstLevel(0);
	extractor_orb_->setWTA_K(2);
	extractor_orb_->setScoreType(cv::ORB::HARRIS_SCORE);
	extractor_orb_->setPatchSize(31);
	extractor_orb_->setFastThreshold(this->THRES_FAST_);

	this->index_bins_.resize(n_bins_u_*n_bins_v_);
	for(int i = 0; i < index_bins_.size(); ++i)
	{
		index_bins_[i].index_.reserve(100);
		index_bins_[i].index_max_ = -1;
		index_bins_[i].max_score_ = -1.0f;
	}

	this->weight_bin_ = std::make_shared<WeightBin>();
	this->weight_bin_->init(n_cols, n_rows, this->n_bins_u_, this->n_bins_v_);
};

void FeatureExtractor::resetWeightBin() {
	// printf(" - FEATURE_EXTRACTOR - 'resetWeightBin'\n");
	weight_bin_->reset();
};

void FeatureExtractor::suppressCenterBins(){
	int u_cent = params_orb_.n_bins_u*0.5;	
	int v_cent = params_orb_.n_bins_v*0.5;


	int win_sz_u = (int)(0.15f*params_orb_.n_bins_u);
	int win_sz_v = (int)(0.30f*params_orb_.n_bins_v);
	int win_sz_v2 = (int)(0.15f*params_orb_.n_bins_v);
	for(int w = -win_sz_v; w <= win_sz_v; ++w){
		int v_idx = params_orb_.n_bins_u*(w+v_cent-win_sz_v2);
		for(int u = -win_sz_u; u <= win_sz_u; ++u){
			int bin_idx = v_idx + u + u_cent;
			// std::cout << bin_idx << std::endl;
			weight_bin_->weight[bin_idx] = 0;
		}
	}
};

void FeatureExtractor::updateWeightBin(const PixelVec& fts) {
	// std::cout << " - FEATURE_EXTRACTOR - 'updateWeightBin'\n";
	weight_bin_->reset();
	weight_bin_->update(fts);
};

void FeatureExtractor::extractORBwithBinning(const cv::Mat& img, PixelVec& pts_extracted, bool flag_nonmax) {
	// INPUT IMAGE MUST BE CV_8UC1 image.

	cv::Mat img_in;
	if (img.type() != CV_8UC1) {
		img.convertTo(img_in, CV_8UC1);
		std::cout << "in extractor, image is converted to the CV_8UC1.\n";
	}
	else 
		img_in = img;

	int n_cols = img_in.cols;
	int n_rows = img_in.rows;

	int overlap = floor(1 * params_orb_.EdgeThreshold);

	std::vector<cv::KeyPoint> fts_tmp;

	pts_extracted.resize(0);
	pts_extracted.reserve(1000);
	fts_tmp.reserve(100);

	const std::vector<int>& u_idx = weight_bin_->u_bound;
	const std::vector<int>& v_idx = weight_bin_->v_bound;

	int v_range[2] = { 0,0 };
	int u_range[2] = { 0,0 };
	for (int v = 0; v < n_bins_v_; ++v) 
	{
		for (int u = 0; u < n_bins_u_; ++u) 
		{
			int bin_idx = v * n_bins_u_ + u;
			int n_pts_desired = weight_bin_->weight[bin_idx] * params_orb_.MaxFeatures;
			if (n_pts_desired > 0) {
				// Set the maximum # of features for this bin.
				// Crop a binned image
				if (v == 0) 
				{
					v_range[0] = v_idx[v];
					v_range[1] = v_idx[v + 1] + overlap;
				}
				else if (v == n_bins_v_ - 1) 
				{
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1];
				}
				else 
				{
					v_range[0] = v_idx[v] - overlap;
					v_range[1] = v_idx[v + 1] + overlap;
				}
				if(v_range[0] <= 0) v_range[0] = 0;
				if(v_range[1] > n_rows) v_range[1] = n_rows-1;

				if (u == 0) {
					u_range[0] = u_idx[u];
					u_range[1] = u_idx[u + 1] + overlap;
				}
				else if (u == n_bins_u_ - 1) {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1];
				}
				else {
					u_range[0] = u_idx[u] - overlap;
					u_range[1] = u_idx[u + 1] + overlap;
				}
				if(u_range[0] <= 0) u_range[0] = 0;
				if(u_range[1] > n_cols) u_range[1] = n_cols-1;
				
				// image sampling
				// TODO: which one is better? : sampling vs. masking
				// std::cout << "set ROI \n";
				cv::Rect roi = cv::Rect(cv::Point(u_range[0], v_range[0]), cv::Point(u_range[1], v_range[1]));
				// std::cout << "roi: " << roi << std::endl;
				// std::cout << "image size : " << img_in.size() <<std::endl;
				
				cv::Mat img_small = img_in(roi);
				fts_tmp.resize(0);
				extractor_orb_->detect(img_small, fts_tmp);

				int n_pts_tmp = fts_tmp.size();
				if (n_pts_tmp > 0) 
				{ 
					//feature can be extracted from this bin.
					int u_offset = 0;
					int v_offset = 0;
					if (v == 0) v_offset = 0;
					else if (v == n_bins_v_ - 1) v_offset = v_idx[v] - overlap - 1;
					else v_offset = v_idx[v] - overlap - 1;
					if (u == 0) u_offset = 0;
					else if (u == n_bins_u_ - 1) u_offset = u_idx[u] - overlap - 1;
					else u_offset = u_idx[u] - overlap - 1;

					std::sort(fts_tmp.begin(), fts_tmp.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
					if (flag_nonmax == true && fts_tmp.size() > NUM_NONMAX_) // select most responsive point in a bin
						fts_tmp.resize(NUM_NONMAX_); // maximum two points.
				
					cv::Point2f pt_offset(u_offset, v_offset);
					for (auto it : fts_tmp) {
						it.pt += pt_offset;
						pts_extracted.push_back(it.pt);
					}
				}
			}
		}
	}

	// Final result
	// std::cout << " - FEATURE_EXTRACTOR - 'extractORBwithBinning' - # detected pts : " << pts_extracted.size() << std::endl;
};


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

	const int n_cols = img_in.cols;
	const int n_rows = img_in.rows;
	const double inv_n_cols = 1.0 / static_cast<double>(n_cols);
	const double inv_n_rows = 1.0 / static_cast<double>(n_rows);

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
	struct IndexAndResponse {
		size_t index;
		float response;
	};

	std::vector<std::vector<IndexAndResponse>> index_and_response_bins(n_total_bins);
	for(size_t index_feature = 0; index_feature < n_pts; ++index_feature) {
		const cv::KeyPoint& kpt = kpts_all[index_feature];

		const double bin_size_column = static_cast<double>(n_cols) / static_cast<double>(n_bins_u);
		const double bin_size_row = static_cast<double>(n_rows) / static_cast<double>(n_bins_v);
		const double inverse_bin_size_column = 1.0 / bin_size_column;
		const double inverse_bin_size_row = 1.0 / bin_size_row;

		const int bin_column = static_cast<int>(kpt.pt.x * inverse_bin_size_column);
		const int bin_row = static_cast<int>(kpt.pt.y * inverse_bin_size_row);

		if(bin_column < 0 || bin_column >= n_bins_u || bin_row < 0 || bin_row >= n_bins_v)
			continue;

		const size_t index_bin = bin_column + n_bins_u * bin_row;
		index_and_response_bins[index_bin].push_back({index_feature, kpt.response});
	}

	// sort and remain fixed number of features per bin
	auto compare_functor = [](const IndexAndResponse& a, const IndexAndResponse& b) {
		return a.response < b.response;
	};
	
	if(n_maximum_feature_per_bin > 1) {
		for(size_t index_bin = 0; index_bin < n_total_bins; ++index_bin) {
			std::vector<IndexAndResponse>& indexes_and_responses = index_and_response_bins[index_bin];

			if(indexes_and_responses.size() == 0) continue;

			if(indexes_and_responses.size() > 1) {
				std::sort(indexes_and_responses.begin(), indexes_and_responses.end(), compare_functor);
				if(indexes_and_responses.size() > n_maximum_feature_per_bin)
					indexes_and_responses.resize(n_maximum_feature_per_bin);
			}
		}
	}

	kpts_extracted.resize(0);
	desc_extracted.resize(0);
	kpts_extracted.reserve(n_total_bins);
	desc_extracted.reserve(n_total_bins);
	for(const std::vector<IndexAndResponse>& indexes_and_responses : index_and_response_bins) {
		if(indexes_and_responses.empty()) continue;

		for(const IndexAndResponse& index_and_response : indexes_and_responses) {
			const cv::KeyPoint& kpt = kpts_all[index_and_response.index];
			const cv::Mat& desc = desc_all[index_and_response.index];
			kpts_extracted.push_back(kpt);
			desc_extracted.push_back(desc);
		}
	}
}

void FeatureExtractor::setNonmaxSuppression(bool flag_on) {
	flag_nonmax_ = true;
};
