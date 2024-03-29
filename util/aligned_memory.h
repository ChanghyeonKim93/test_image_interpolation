#ifndef _ALIGNED_MEMORY_H_
#define _ALIGNED_MEMORY_H_
#include <iostream>
#include <memory>

// Ref:
// https://codereview.stackexchange.com/questions/90758/implementation-of-unique-ptr-and-make-unique-for-aligned-memory

#define ALIGN_BYTES 64 // AVX, AVX2 (256 bits = 32 Bytes), SSE4.2 (128 bits = 16 Bytes)
/** \internal Like malloc, but the returned pointer is guaranteed to be 32-byte aligned.
 * Fast, but wastes 32 additional bytes of memory. Does not throw any exception.
 *
 * (256 bits) two LSB addresses of 32 bytes-aligned : 00, 20, 40, 60, 80, A0, C0, E0
 * (128 bits) two LSB addresses of 16 bytes-aligned : 00, 10, 20, 30, 40, 50, 60, 70, 80, 90, A0,
 * B0, C0, D0, E0, F0
 */
namespace aligned_memory {
template <typename T> inline T *malloc(size_t length) {
  const size_t size = length * sizeof(T);
  void *original = std::malloc(size + ALIGN_BYTES); // size+ALIGN_BYTES��ŭ �Ҵ��ϰ�,
  if (original == 0)
    return nullptr; // if allocation is failed, return nullptr;
  void *aligned = reinterpret_cast<void *>(
    (reinterpret_cast<std::size_t>(original) & ~(std::size_t(ALIGN_BYTES - 1))) + ALIGN_BYTES);
  *(reinterpret_cast<void **>(aligned) - 1) = original;
  return (T *)aligned;
};

/** \internal Frees memory allocated with handmade_aligned_malloc */
template <typename T> inline void free(T *ptr) {
  if (ptr != nullptr)
    std::free(*(reinterpret_cast<void **>(ptr) - 1));
};
}; // namespace aligned_memory

#endif