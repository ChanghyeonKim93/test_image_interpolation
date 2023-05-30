#ifndef _PRIORITY_QUEUE_FIXED_SIZE_H_
#define _PRIORITY_QUEUE_FIXED_SIZE_H_
#include <iostream>

template<typename T, typename Compare=std::less<T>>
class PriorityQueueFixedSize
{
  public:
    PriorityQueueFixedSize() :max_size_(0) {}
    PriorityQueueFixedSize(size_t max_size) : max_size_(max_size) {}

    typedef typename std::vector<T>::iterator iterator;
    iterator begin() { return elements_.begin(); }
    iterator end() { return elements_.end(); }

    inline void push(const T &x) {
      if(elements_.size() == max_size_) {
        // 현재까지의 최소값을 가져온다.
        typename std::vector<T>::iterator iterator_min = 
          std::min_element(elements_.begin(), elements_.end(), cmp_);
        
        // 최소치보다 x가 크면 최소치와 바꿔주고 시작한다.
        if(*iterator_min < x) {
          *iterator_min = x;
          std::make_heap(elements_.begin(), elements_.end(), cmp_);
        }
      }
      else {
        elements_.push_back(x);
        std::make_heap(elements_.begin(), elements_.end(), cmp_);
      }
    }
    
  protected:
    std::vector<T> elements_;
    size_t max_size_;
    Compare cmp_;
};

#endif