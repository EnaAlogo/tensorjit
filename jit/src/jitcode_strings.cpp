#include  "jitcode_strings.h"


//this file uses the int64 divisor from cutlass and some type casts from pytorch 

/*
* From cutlass 
* 
Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
* From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/
namespace megu
{
    const std::string preamble_code =
        R"ESC(
//maybe bake this in ?? will possibly increase compiled kernels so no
#define MAX_DIMS 12
typedef long long int int64_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;  
typedef unsigned short uint16_t;
typedef int int32_t;
typedef short int16_t;
typedef unsigned long long int uint64_t;
static_assert(sizeof(int64_t) == 8, "expected size does not match");
static_assert(sizeof(uint32_t) == 4, "expected size does not match");
static_assert(sizeof(int8_t) == 1, "expected size does not match");

__device__ __forceinline__ float rsqrt(float x){return ::__frsqrt_rn(x);};
__device__ __forceinline__ double rsqrt(double x){return ::rsqrt(x);};
template<typename T>
__device__ __forceinline__ T conj(T val){return val;}
namespace megu{
template<typename T>
struct acc_type{
using type = T;
};
template<typename T>
using acctype_t = typename acc_type<T>::type;
}//end megu

#if !defined(JITIFY2)
namespace std{
template< class... >
using void_t = void;
}//end std
#endif

namespace megu {namespace detail{
template<typename T, int Size>
struct Carray
{
	__device__ __forceinline__ void fill(T val) { 
#pragma unroll
		for (int i = 0; i < Size; ++i)
			data_[i] = val;
	}

	__device__ __forceinline__ T& operator[](int offset) { return data_[offset]; };
	__device__ __forceinline__ T operator[](int offset) const { return data_[offset]; };
	__device__ static constexpr int size() { return Size; };
	T data_[Size]; 

};
	
}}//end megu::detail

#include <type_traits>

#define typeof(x) std::remove_const_t<std::remove_reference_t<decltype(x)>>

namespace megu{
template<typename T , typename std::enable_if_t<std::is_integral<T>::value,int> = 0>
inline bool IsNan(T val) {
  return false;
}
inline bool IsNan(double val) {
  return isnan(val);
}
inline bool IsNan(float val) {
  return isnan(val);
}
}//end megu

)ESC";


    const std::string float16_code =
        R"ESC(
#ifndef MEGU_FP16_H
#define MEGU_FP16_H
#endif

namespace megu{
struct alignas(2) float16 {
  unsigned short x;

  float16() = default;

  inline __host__ __device__ float16(float value){
#ifdef __HIPCC__
    x = __half_as_short(__float2half(value));
#else
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
#endif
  }
  inline __host__ __device__ operator float() const{
#ifdef __HIPCC__
      return __half2float(*reinterpret_cast<const __half*>(&x));
#else
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(x)); 
      //asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(x)));
      return val;
#endif
  }
#if defined(__CUDA_FP16_H__)
  inline __host__ __device__  float16(const __half& value) {
      x = *reinterpret_cast<const uint16_t*>(&value);
  }
  inline __host__ __device__  operator __half() const {
      return *reinterpret_cast<const __half*>(&x);
  }
#endif
};
template<>
struct acc_type<float16>{using type = float;};
}//end megu
)ESC";

    const std::string float16_math_code = R"ESC( 
namespace megu{

inline bool IsNan(megu::float16 val) {
    return isnan(static_cast<float>(val));
}

inline float16 operator+(const float16& a, const float16& b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
        return __hadd(static_cast<__half>(a),static_cast<__half>(b));
#else
        return static_cast<float>(a) + static_cast<float>(b);
#endif
    }

    inline float16 operator-(const float16& a, const float16& b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
        return __hsub(a, b);
#else
        return static_cast<float>(a) - static_cast<float>(b);
#endif
    }

    inline float16 operator*(const float16& a, const float16& b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
        return __hmul(a, b);
#else
        return static_cast<float>(a) * static_cast<float>(b);
#endif
    }

    inline float16 operator/(const float16& a, const float16& b)
    {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
        return __hdiv(a, b);
#else
        return static_cast<float>(a) / static_cast<float>(b);
#endif
    }

    inline float16 operator-(const float16& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
        return __hneg(a);
#else
        return -static_cast<float>(a);
#endif
    }

    inline float16& operator+=(float16& a, const float16& b) { 
        a = a + b;
        return a;
    }

    inline float16& operator-=(float16& a, const float16& b) {
        a = a - b;
        return a;
    }

    inline float16& operator*=(float16& a, const float16& b) {
        a = a * b;
        return a;
    }

    inline float16& operator/=(float16& a, const float16& b) {
        a = a / b;
        return a;
    }

    /// Arithmetic with floats

    inline float operator+(float16 a, float b) {
        return static_cast<float>(a) + b;
    }
    inline float operator-(float16 a, float b) {
        return static_cast<float>(a) - b;
    }
    inline float operator*(float16 a, float b) {
        return static_cast<float>(a) * b;
    }
    inline float operator/(float16 a, float b)
    {
        return static_cast<float>(a) / b;
    }

    inline float operator+(float a, float16 b) {
        return a + static_cast<float>(b);
    }
    inline float operator-(float a, float16 b) {
        return a - static_cast<float>(b);
    }
    inline float operator*(float a, float16 b) {
        return a * static_cast<float>(b);
    }
    inline float operator/(float a, float16 b)
    {
        return a / static_cast<float>(b);
    }

    inline double operator+(float16 a, double b) {
        return static_cast<double>(a) + b;
    }
    inline double operator-(float16 a, double b) {
        return static_cast<double>(a) - b;
    }
    inline double operator*(float16 a, double b) {
        return static_cast<double>(a) * b;
    }
    inline double operator/(float16 a, double b)
    {
        return static_cast<double>(a) / b;
    }

    inline double operator+(double a, float16 b) {
        return a + static_cast<double>(b);
    }
    inline double operator-(double a, float16 b) {
        return a - static_cast<double>(b);
    }
    inline double operator*(double a, float16 b) {
        return a * static_cast<double>(b);
    }
    inline double operator/(double a, float16 b)
    {
        return a / static_cast<double>(b);
    }

    inline float16 operator+(float16 a, int b) {
        return a + static_cast<float16>(b);
    }
    inline float16 operator-(float16 a, int b) {
        return a - static_cast<float16>(b);
    }
    inline float16 operator*(float16 a, int b) {
        return a * static_cast<float16>(b);
    }
    inline float16 operator/(float16 a, int b) {
        return a / static_cast<float16>(b);
    }

    inline float16 operator+(int a, float16 b) {
        return static_cast<float16>(a) + b;
    }
    inline float16 operator-(int a, float16 b) {
        return static_cast<float16>(a) - b;
    }
    inline float16 operator*(int a, float16 b) {
        return static_cast<float16>(a) * b;
    }
    inline  float16 operator/(int a, float16 b) {
        return static_cast<float16>(a) / b;
    }
}//end megu
)ESC";

    const std::string bfloat_code =
        R"ESC(
#ifndef MEGU_BP16_H
#define MEGU_BP16_H
#endif

namespace megu{
struct alignas(2) bfloat16 {
  unsigned short x;

  __device__ unsigned short __internal_float2bfloat16(
      const float f,
      unsigned int& sign,
      unsigned int& remainder) {
    unsigned int x;

    x = __float_as_uint(f);

    if ((x & 0x7fffffffU) > 0x7f800000U) {
      sign = 0U;
      remainder = 0U;
      return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31;
    remainder = x << 16;
    return static_cast<unsigned short>(x >> 16);
  }


  bfloat16() = default;
  inline __host__ __device__ bfloat16(float value){
  #if __CUDA_ARCH__ >= 800
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(x) : "f"(value));
  #else
  unsigned int sign;
  unsigned int remainder;
  x = __internal_float2bfloat16(value, sign, remainder);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((x & 0x1U) != 0U))) {
    x++;
  }
  #endif
  }

  inline __host__ __device__ operator float() const{
#ifdef __HIPCC__
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
#else
    float val;
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(x)); 
    return val;
#endif
  }
#ifdef __CUDA_BF16_H__
  inline bfloat16(const __nv_bfloat16& value) {
      x = *reinterpret_cast<const unsigned short*>(&value);
  }
  inline operator __nv_bfloat16() const {
      return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif
};
template<>
struct acc_type<bfloat16>{using type = float;};
}//end megu
)ESC";

    const std::string bfloat16_math_code = R"ESC(
namespace megu{

inline bool IsNan(megu::bfloat16 val) {
    return isnan(static_cast<float>(val));
}

inline bfloat16
operator+(const bfloat16 & a, const bfloat16 & b) {
#if defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return static_cast<float>(a) + static_cast<float>(b);
#endif
}
inline bfloat16
operator-(const bfloat16 & a, const bfloat16 & b) {
#if defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    return __hsub(a, b);
#else
    return static_cast<float>(a) - static_cast<float>(b);
#endif
}
inline bfloat16
operator*(const bfloat16 & a, const bfloat16 & b) {
#if defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return static_cast<float>(a) * static_cast<float>(b);
#endif
}
inline bfloat16 operator/(const bfloat16 & a, const bfloat16 & b) {
#if defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    return __hdiv(a, b);
#else
    return static_cast<float>(a) / static_cast<float>(b);
#endif
}
inline bfloat16 operator-(const bfloat16 & a) {
#if defined(__CUDACC__) && !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    return __hneg(a);
#else
    return -static_cast<float>(a);
#endif
}
inline bfloat16& operator+=(bfloat16 & a, const bfloat16 & b) {
    a = a + b;
    return a;
}
inline bfloat16& operator-=(bfloat16 & a, const bfloat16 & b) {
    a = a - b;
    return a;
}
inline bfloat16& operator*=(bfloat16 & a, const bfloat16 & b) {
    a = a * b;
    return a;
}
inline bfloat16& operator/=(bfloat16 & a, const bfloat16 & b) {
    a = a / b;
    return a;
}
inline bfloat16& operator|(bfloat16 & a, const bfloat16 & b) {
    a.x = a.x | b.x;
    return a;
}
inline bfloat16& operator^(bfloat16 & a, const bfloat16 & b) {
    a.x = a.x ^ b.x;
    return a;
}
inline bfloat16& operator&(bfloat16 & a, const bfloat16 & b) {
    a.x = a.x & b.x;
    return a;
}
inline float operator+(bfloat16 a, float b) {
    return static_cast<float>(a) + b;
}
inline float operator-(bfloat16 a, float b) {
    return static_cast<float>(a) - b;
}
inline float operator*(bfloat16 a, float b) {
    return static_cast<float>(a) * b;
}
inline float operator/(bfloat16 a, float b) {
    return static_cast<float>(a) / b;
}
inline float operator+(float a, bfloat16 b) {
    return a + static_cast<float>(b);
}
inline float operator-(float a, bfloat16 b) {
    return a - static_cast<float>(b);
}
inline float operator*(float a, bfloat16 b) {
    return a * static_cast<float>(b);
}
inline float operator/(float a, bfloat16 b) {
    return a / static_cast<float>(b);
}
inline double operator+(bfloat16 a, double b) {
    return static_cast<double>(a) + b;
}
inline double operator-(bfloat16 a, double b) {
    return static_cast<double>(a) - b;
}
inline double operator*(bfloat16 a, double b) {
    return static_cast<double>(a) * b;
}
inline double operator/(bfloat16 a, double b) {
    return static_cast<double>(a) / b;
}
inline double operator+(double a, bfloat16 b) {
    return a + static_cast<double>(b);
}
inline double operator-(double a, bfloat16 b) {
    return a - static_cast<double>(b);
}
inline double operator*(double a, bfloat16 b) {
    return a * static_cast<double>(b);
}
inline double operator/(double a, bfloat16 b) {
    return a / static_cast<double>(b);
}
inline bfloat16 operator+(bfloat16 a, int b) {
    return a + static_cast<bfloat16>(b);
}
inline bfloat16 operator-(bfloat16 a, int b) {
    return a - static_cast<bfloat16>(b);
}
inline bfloat16 operator*(bfloat16 a, int b) {
    return a * static_cast<bfloat16>(b);
}
inline bfloat16 operator/(bfloat16 a, int b) {
    return a / static_cast<bfloat16>(b);
}

inline bfloat16 operator+(int a, bfloat16 b) {
    return static_cast<bfloat16>(a) + b;
}
inline bfloat16 operator-(int a, bfloat16 b) {
    return static_cast<bfloat16>(a) - b;
}
inline bfloat16 operator*(int a, bfloat16 b) {
    return static_cast<bfloat16>(a) * b;
}
inline bfloat16 operator/(int a, bfloat16 b) {
    return static_cast<bfloat16>(a) / b;
}
}//end megu)ESC";

    const std::string indexer_code =

        R"esc( 
namespace megu{
template <typename T>
  struct DivMod {
  T div;
  T mod;

  __device__ DivMod(T _div, T _mod) {
      div = _div;
      mod = _mod;
  }
  };

template<typename T>
struct IntDivider{
  T dvd;
  IntDivider() = default;
   __device__ inline unsigned int div(unsigned int n) const {
       return n / dvd;
   }
   
   __device__ inline unsigned int mod(unsigned int n) const {
       return n % dvd;
   }
   
   __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
      unsigned int q = div(n);
      return DivMod<unsigned int>(q, n - q * dvd);
   }

};
template<>
struct IntDivider<unsigned int> {
   IntDivider() = default;
   
   __device__ inline unsigned int div(unsigned int n) const {
   unsigned int t = __umulhi(n, m1);
   return (t + n) >> shift;
   }
   
   __device__ inline unsigned int mod(unsigned int n) const {
   return n - div(n) * divisor;
   }
   
   __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
   unsigned int q = div(n);
   return DivMod<unsigned int>(q, n - q * divisor);
   }
   
   unsigned int divisor;  // d above.
   unsigned int m1;  // Magic number: m' above.
   unsigned int shift;  // Shift amounts.
};
template <>
struct IntDivider<uint64_t> {//from cutlass project see fast_divmod.h 
	uint64_t divisor;
	uint64_t multiplier;
	unsigned int shift_right;
	unsigned int round_up;

	IntDivider() = default;

	__device__ inline uint64_t div(uint64_t dividend) const {
		uint64_t quotient = 0;

		uint64_t x = dividend;
		if (multiplier) {
			x = __umul64hi(dividend + round_up, multiplier);
		}
		quotient = (x >> shift_right);

		return quotient;
	}

	/// Computes the remainder given a computed quotient and dividend
	__device__ inline uint64_t mod(uint64_t quotient, uint64_t dividend) const {
		return uint32_t(dividend - quotient * divisor);
	}

	/// Returns the quotient of floor(dividend / divisor) and computes the remainder
	__device__ inline DivMod<uint64_t> divmod(uint64_t dividend) const {
		uint64_t quotient = div(dividend);
		uint64_t remainder = dividend - quotient * divisor;
		return { quotient, remainder };
	}
};
}//end megu 
namespace megu::detail{
template<int argc , typename index_t = unsigned int , bool allow_signed = false>
  struct ContiguousIndexToOffset {
    __device__ __forceinline__ Carray<index_t, (argc > 0 ? argc : 1)> get(index_t linear_idx) const {
      Carray<index_t, (argc > 0 ? argc : 1)> offsets;
      #pragma unroll
      for (int arg = 0; arg < argc; arg++) {
        offsets[arg] = linear_idx;
      }
      return offsets;
    }
  };

template<int argc , typename index_t = unsigned int, bool alllow_signed = false>
  struct IndexToOffset {
  IndexToOffset() = default;
  __device__ __forceinline__ Carray<index_t, (argc > 0 ? argc : 1)> get(index_t linear_idx) const {
      Carray<index_t, (argc > 0 ? argc : 1)> offsets{0};

      #pragma unroll
      for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
          break;
      }

      auto divmod = shape[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < argc; ++arg) {
          offsets[arg] += divmod.mod * strides_[dim][arg];
      }
      }
      return offsets;
  }

    	IntDivider<index_t> shape[MAX_DIMS]; 
		index_t strides_[MAX_DIMS][ (argc > 0 ? argc : 1) ];
		int8_t dims;
  };
}//end megu::detail
)esc";

}//end megu