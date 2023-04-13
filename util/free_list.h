#ifndef _FREE_LIST_H_
#define _FREE_LIST_H_
#include <iostream>

template<typename T>
class FreeList
{
	FreeList() 
		: ptr_head_(nullptr)
	{

	}

	T* getObject()
	{
		if (ptr_head_ == nullptr)
		{
			// free list에 없으면 heap에서 할당한 객체를 리턴합니다.
			Item* item_new = new Item();
			item_new->ptr_next_ = ptr_head_;
			ptr_head_ = item_new;
			return &item_new->object;
		}
		else
		{
			// free list에 있으면 head에서 꺼내서 리턴합니다
			T* object_old = ptr_head_;
			ptr_head_ = ptr_head_->ptr_next_;
			return &object_old->object;
		}
	}

	void returnObject(T* object)
	{
		// 사용자가 사용을 다 마친 객체를 반환하는 것은 간단합니다. 그냥 free list에 넣읍시다.
		// 사용자의 객체는 FreeList::Item 안에 있던 객체의 주소입니다.
		// 이것을 소유한 FreeList::Item 자체의 주소를 얻어 옵시다.
		Item* listItem = (Item*)(((char*)object) - offsetof(Item, m_item));
		listItem->ptr_next_ = ptr_head_;
		ptr_head_ = listItem;
	}

private:
	struct Item
	{
		T object; // 객체
		Item* ptr_next_; // 다음 free list를 가리킨다.
	};

	T* ptr_head_; // 첫 객체
};


#endif
