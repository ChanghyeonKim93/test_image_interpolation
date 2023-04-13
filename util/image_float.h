#ifndef _IMAGE_FLOAT_H_
#define _IMAGE_FLOAT_H_

#include <iostream>
#include "aligned_memory.h"

template <typename _data_type>
class Image
{
public:
  Image()
      : cols_(0), rows_(0), n_elem_(cols_ * rows_), data_(nullptr)
  {
	// default constructor
  }
  Image(const size_t n_rows, const size_t n_cols)
      : cols_(n_cols), rows_(n_rows), n_elem_(cols_ * rows_), data_(nullptr)
  {
    data_ = aligned_memory::malloc<_data_type>(n_elem_);
  }
  ~Image()
  {
    if (!data_)
      aligned_memory::free<_data_type>(data_);
  }

public:
  _data_type *data() const { return data_; };
  const size_t col() const { return cols_; };
  const size_t row() const { return rows_; };
  void fillZero()
  {
    if (!data_)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = 0.0f;
    }
  }
  void fillOne()
  {
    if (!data_)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = 1.0f;
    }
  }
  void fill(const _data_type value)
  {
    if (!data_)
    {
      _data_type *p = data_;
      const _data_type *p_end = data_ + n_elem_;
      for (; p != p_end; ++p)
        *p = value;
    }
  }

public:
  void copyTo(Image<_data_type> &dst)
  {
    dst.cols_ = this->cols_;
    dst.rows_ = this->rows_;
    dst.n_elem_ = this->n_elem_;
    if (!dst.data_)
      aligned_memory::free<_data_type>(dst.data_);
    dst.data_ = aligned_memory::malloc<_data_type>(n_elem_);

    _data_type *p_src = this->data_;
    _data_type *p_dst = dst.data_;
    const _data_type *p_src_end = this->data_ + this->n_elem_;
    for (; p_src != p_src_end; ++p_src, ++p_dst)
      *p_dst = *p_src;
  }

public:
  // operator=
  // operator+
  // operator-
	Image<_data_type>& operator=(const Image<_data_type>& input)
	{
		// If the size is same && data_ is allocated, just copy.
		if (this->cols_ == input.cols_ && this->rows_ == input.rows_ && this->n_elem_ == input.n_elem_)
		{
			if (!data_)
				data_ = aligned_memory::malloc<_data_type>(n_elem_);
			else
				memcpy(this->data_, input.data_, sizeof(float)*this->n_elem_);
		}
		else
		{
			this->cols_ = input.cols_;
			this->rows_ = input.rows_;
			this->n_elem_ = input.n_elem_;
			if (!data_)
			{
				aligned_memory::free(data_);
				data_ = aligned_memory::malloc<_data_type>(n_elem_);
				memcpy(this->data_, input.data_, sizeof(float)*this->n_elem_);
			}
		}
		return *this;
	}
	/*Image<_data_type> operator+(const Image<_data_type>& input)
	{
		Image<_data_type> dst(input.row, input.col);

		return 
	}
	Image<_data_type> operator-(const Image<_data_type>& input)
	{
		return
	}*/

private:
  size_t cols_;
  size_t rows_;
  size_t n_elem_;
  _data_type *data_;
};

#endif