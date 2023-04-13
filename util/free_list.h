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
			// free list�� ������ heap���� �Ҵ��� ��ü�� �����մϴ�.
			Item* item_new = new Item();
			item_new->ptr_next_ = ptr_head_;
			ptr_head_ = item_new;
			return &item_new->object;
		}
		else
		{
			// free list�� ������ head���� ������ �����մϴ�
			T* object_old = ptr_head_;
			ptr_head_ = ptr_head_->ptr_next_;
			return &object_old->object;
		}
	}

	void returnObject(T* object)
	{
		// ����ڰ� ����� �� ��ģ ��ü�� ��ȯ�ϴ� ���� �����մϴ�. �׳� free list�� �����ô�.
		// ������� ��ü�� FreeList::Item �ȿ� �ִ� ��ü�� �ּ��Դϴ�.
		// �̰��� ������ FreeList::Item ��ü�� �ּҸ� ��� �ɽô�.
		Item* listItem = (Item*)(((char*)object) - offsetof(Item, m_item));
		listItem->ptr_next_ = ptr_head_;
		ptr_head_ = listItem;
	}

private:
	struct Item
	{
		T object; // ��ü
		Item* ptr_next_; // ���� free list�� ����Ų��.
	};

	T* ptr_head_; // ù ��ü
};


#endif
