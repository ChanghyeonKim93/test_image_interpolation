#ifndef _FREE_LIST_H_
#define _FREE_LIST_H_
#include <iostream>

template <typename T>
class FreeList
{
	FreeList()
			: ptr_head_(nullptr), num_of_available_object_(0), num_of_total_object_(0)
	{
	}

	T *getObject()
	{
		ObjectWrapper *ow_new;
		if (ptr_head_ == nullptr)
		{
			++num_of_total_object_;
			ow_new = new ObjectWrapper(); 
			ow_new->ptr_next_ = nullptr;
			ptr_head_ = ow_new;
		}
		else
		{
			ObjectWrapper *ow_new = ptr_head_;
			ptr_head_ = ow_new->ptr_next_;
		}
		return ow_new->object;
	}

	void returnObject(T *object)
	{
		ObjectWrapper *listItem = (Item *)(((char *)object) - offsetof(Item, m_item));
		listItem->ptr_next_ = ptr_head_;
		ptr_head_ = listItem;
	}

private:
	class ObjectWrapper
	{
	public:
		ObjectWrapper() { object = std::shared_ptr<T>(); } // default constructor is needed.
		~ObjectWrapper() {}

	private:
		T *object;								//
		ObjectWrapper *ptr_next_; // free list
	};

	ObjectWrapper *ptr_head_; // The first list.

	size_t num_of_total_object_;
	size_t num_of_available_object_;
};

#endif
