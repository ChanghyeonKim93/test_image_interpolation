#ifndef _IMAGE_FLOAT_H_
#define _IMAGE_FLOAT_H_

#include <iostream>
#include "aligned_memory.h"

// - Image class. It allocates 'aligned memory'.
template <typename _data_type>
class Image
{
public:
  Image()
      : rows_(0), cols_(0), n_elem_(cols_ * rows_), data_(nullptr)
  {
  }
  Image(const size_t n_rows, const size_t n_cols)
    : rows_(n_rows), cols_(n_cols), n_elem_(cols_ * rows_), data_(nullptr)
  {
    data_ = aligned_memory::malloc<_data_type>(n_elem_);
		this->fillZero();
  }
	Image(const Image& rhs)
		: rows_(rhs.row()), cols_(rhs.col()), n_elem_(cols_ * rows_), data_(nullptr)
	{
		// Assign constructor
		data_ = aligned_memory::malloc<_data_type>(n_elem_);
		memcpy(data_, rhs.data_, n_elem_ * sizeof(_data_type));
	}
  ~Image()
  {
		rows_ = 0;
		cols_ = 0;
		n_elem_ = 0;
    if (data_ != nullptr)
      aligned_memory::free<_data_type>(data_);
  }

public:
	_data_type *data() const { return data_; };
	_data_type *getPtr(const size_t row_number) const 
	{ 
		if (row_number >= rows_) {
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "row_number >= rows_");
			return nullptr;
		}
		return (data_ + row_number * cols_);
	}
	_data_type *getPtr(const size_t row_number, const size_t col_number) const 
	{
		if(row_number >= rows_)
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "row_number >= rows_");
		if (col_number >= cols_)
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "col_number >= cols_");

		const size_t index = row_number * cols_ + col_number;
		if (index >= n_elem_) {
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "index >=  n_elem_");
			return nullptr;
		}
		return (data_ + index);
	}
  const size_t col() const { return cols_; };
  const size_t row() const { return rows_; };
	
public:
  void fillZero()
  {
    if (data_ != nullptr)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = 0.0f;
    }
  }
  void fillOne()
  {
    if (data_ != nullptr)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = 1.0f;
    }
  }
  void fill(const _data_type value)
  {
    if (data_ != nullptr)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = value;
    }
  }

public:
  void copyTo(Image<_data_type> &dst) const
  {
    dst.cols_ = this->cols_;
    dst.rows_ = this->rows_;
    dst.n_elem_ = this->n_elem_;
    if (dst.data_ != nullptr)
      aligned_memory::free<_data_type>(dst.data_);
    dst.data_ = aligned_memory::malloc<_data_type>(n_elem_);

		memcpy(dst.data_, this->data_, sizeof(_data_type)*this->n_elem_);
  }

public:
	// 기존 데이터 존재
	// ..대입 이미지가 같은 크기 : 데이터 복사
	// ..대입 이미지가 다른 크기 : free -> malloc -> 데이터 복사
	// ....대입 이미지 크기 < 피대입 이미지 크기: 공간 전체 n_cols, n_rows 
	// ....대입 이미지가 큰 경우   : 
	// 기존 데이터 없음
	// ..malloc -> 데이터 복사
	Image<_data_type>& operator=(const Image<_data_type>& rhs)
	{

		if (this == &rhs) return *this;	// 자기 대입인지를 검사
		if (this->cols_ == rhs.cols_ && this->rows_ == rhs.rows_ && this->n_elem_ == rhs.n_elem_)
		{
			// If the size is same && data_ is allocated, just copy.
			if (data_ == nullptr)
				data_ = aligned_memory::malloc<_data_type>(n_elem_);
			else
				memcpy(this->data_, rhs.data_, sizeof(float)*this->n_elem_);
		}
		else
		{
			this->rows_ = rhs.rows_;
			this->cols_ = rhs.cols_;
			this->n_elem_ = this->cols_ * this->rows_;

			if (data_ != nullptr)
			{
				aligned_memory::free(data_); 
				data_ = nullptr;
			}
			this->data_ = aligned_memory::malloc<_data_type>(n_elem_);
			memcpy(this->data_, rhs.data_, sizeof(float)*this->n_elem_);
		}
		return *this;
	}

	Image<_data_type> operator+(const Image<_data_type>& input)
	{
		Image<_data_type> dst;
		if (this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)
		{
			this->copyTo(dst);

			const _data_type* ptr_input = input.data_;
			_data_type* ptr_dst = dst.data_;
			const _data_type* ptr_dst_end = dst.data_ + dst.n_elem_;
			for (; ptr_dst < ptr_dst_end; ++ptr_dst, ++ptr_input)
			{
				*ptr_dst += *ptr_input;
			}
		}
		else
		{
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "!(this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)");
		}
		return dst;
	}

	Image<_data_type> operator-(const Image<_data_type>& input)
	{
		Image<_data_type> dst;
		if (this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)
		{
			this->copyTo(dst);

			_data_type* ptr_dst = dst.data_;
			const _data_type* ptr_dst_end = dst.data_ + dst.n_elem_;
			const _data_type* ptr_input = input.data_;
			for (; ptr_dst < ptr_dst_end; ++ptr_dst, ++ptr_input)
			{
				*ptr_dst -= *ptr_input;
			}
		}
		else
		{
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "!(this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)");
		}
		return dst;
	}

	Image<_data_type>& operator+=(const Image<_data_type>& input)
	{
		if (this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)
		{
			_data_type* ptr_dst = this->data_;
			const _data_type* ptr_dst_end = this->data_ + this->n_elem_;
			const _data_type* ptr_input = input.data_;
			for (; ptr_dst < ptr_dst_end; ++ptr_dst, ++ptr_input)
			{
				*ptr_dst += *ptr_input;
			}
		}
		else
		{
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "!(this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)");
		}
		return *this;
	}

	Image<_data_type>& operator-=(const Image<_data_type>& input)
	{
		if (this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)
		{
			_data_type* ptr_dst = this->data_;
			const _data_type* ptr_dst_end = this->data_ + this->n_elem_;
			const _data_type* ptr_input = input.data_;
			for (; ptr_dst < ptr_dst_end; ++ptr_dst, ++ptr_input)
			{
				*ptr_dst -= *ptr_input;
			}
		}
		else
		{
			throw std::runtime_error("In function " + std::string(__func__) + ", " + "!(this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)");
		}
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& output_stream, const Image<_data_type>& img) 
	{	
		if (img.n_elem_ == 0)
		{
			std::cout << "[]\n";
		}
		else 
		{
			const _data_type* ptr = img.data_;
			for (size_t v = 0; v < img.row(); ++v)
			{
				std::cout << "[";
				for (size_t u = 0; u < img.col() - 1; ++u)
				{
					std::cout << *ptr << ",";
					++ptr;
				}
				std::cout << *ptr; ++ptr;
				std::cout << "]\n";
			}
		}
		return output_stream;
	}

private:
  size_t cols_;
  size_t rows_;
  size_t n_elem_;
  _data_type *data_;
};

#endif