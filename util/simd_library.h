#ifndef _SIMD_LIBRARY_H_
#define _SIMD_LIBRARY_H_
#define USE_AVX
/*
ps: float
pd: double
epi8/16/32/64: 8,16,32,64 bit signed integer
si128/256: typeless
*/

#ifdef USE_SSE
#define SIMD_STEPSIZE_FLOAT 4
#define SIMD_DATA_FLOAT __m128
#define SIMD_LOAD_FLOAT _mm_load_ps
#define SIMD_STORE_FLOAT _mm_store_ps
#define SIMD_STREAM_FLOAT _mm_stream_ps
#define SIMD_SET1_FLOAT _mm_set1_ps
#define SIMD_ADD_FLOAT _mm_add_ps
#define SIMD_SUB_FLOAT _mm_sub_ps
#define SIMD_MUL_FLOAT _mm_mul_ps
#define SIMD_DIV_FLOAT _mm_div_ps
#define SIMD_RCP_FLOAT _mm_rcp_ps

#define SIMD_STEPSIZE_SHORT 8
#define SIMD_DATA_SHORT __m128i
#define SIMD_LOAD_SHORT _mm_load_si128
#define SIMD_STORE_SHORT _mm_store_si128 // __m128i _mm_loadu_si16 (void const* mem_addr)
#define SIMD_STREAM_SHORT _mm_stream_si128
#define SIMD_SET1_SHORT _mm_set1_epi16
#define SIMD_ADD_SHORT _mm_add_epi16
// #define SIMD_ADDS_SHORT _mm_adds_epi16
#define SIMD_SUB_SHORT _mm_sub_epi16
// #define SIMD_SUBS_SHORT _mm_subs_epi16
#define SIMD_MULLO_SHORT _mm_mullo_epi16
#endif

#ifdef USE_AVX
#define SIMD_STEPSIZE_FLOAT 8
#define SIMD_DATA_FLOAT __m256
#define SIMD_STREAM_FLOAT _mm256_stream_ps
#define SIMD_SET1_FLOAT _mm256_set1_ps
#define SIMD_LOAD_FLOAT _mm256_load_ps
#define SIMD_STORE_FLOAT _mm256_store_ps
#define SIMD_ADD_FLOAT _mm256_add_ps
#define SIMD_SUB_FLOAT _mm256_sub_ps
#define SIMD_MUL_FLOAT _mm256_mul_ps
#define SIMD_DIV_FLOAT _mm256_div_ps
#define SIMD_RCP_FLOAT _mm256_rcp_ps

#define SIMD_STEPSIZE_SHORT 16
#define SIMD_DATA_SHORT __m256i
#define SIMD_LOAD_SHORT _mm256_load_si256
#define SIMD_STORE_SHORT _mm256_store_si256 // __m128i _mm_loadu_si16 (void const* mem_addr)
#define SIMD_STREAM_SHORT _mm256_stream_si256
#define SIMD_SET1_SHORT _mm256_set1_epi16
#define SIMD_ADD_SHORT _mm256_add_epi16
// #define SIMD_ADDS_SHORT _mm_adds_epi16
#define SIMD_SUB_SHORT _mm256_sub_epi16
// #define SIMD_SUBS_SHORT _mm_subs_epi16
#define SIMD_MULLO_SHORT _mm256_mullo_epi16
#endif

#endif