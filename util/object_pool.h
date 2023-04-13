#ifndef _OBJECT_POOL_H_
#define _OBJECT_POOL_H_
#include <iostream>
#include <memory>
#include "aligned_memory.h"

template <typename T>
class ObjectPool 
{
public:
	ObjectPool(const size_t num_of_objects)
	{
		memory_chunk_ = aligned_memory::malloc<T>(num_of_objects);
		if (memory_chunk_ == nullptr)
			throw std::runtime_error("In function " + std::string(__func__) + ", memory_chunk_ == nullptr. Memory allocation is failed.");
	}
	~ObjectPool()
	{
		if (memory_chunk_ != nullptr)
			aligned_memory::free<T>(memory_chunk_);
	}

private:
	T* memory_chunk_;
	size_t capacity_; // maximum # of objects
};

#endif