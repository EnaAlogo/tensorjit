#pragma once
#include <string>
#include <format>


/*
* this is pytorch's jiterator reduction but its been modified 
* for some added features 
*/

/*
From PyTorch :

Copyright(c) 2016 - Facebook, Inc(Adam Paszke)
Copyright(c) 2014 - Facebook, Inc(Soumith Chintala)
Copyright(c) 2011 - 2014 Idiap Research Institute(Ronan Collobert)
Copyright(c) 2012 - 2014 Deepmind Technologies(Koray Kavukcuoglu)
Copyright(c) 2011 - 2012 NEC Laboratories America(Koray Kavukcuoglu)
Copyright(c) 2011 - 2013 NYU(Clement Farabet)
Copyright(c) 2006 - 2010 NEC Laboratories America(Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright(c) 2006      Idiap Research Institute(Samy Bengio)
Copyright(c) 2001 - 2004 Idiap Research Institute(Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2 :

Copyright(c) 2016 - present, Facebook Inc.All rights reserved.

All contributions by Facebook :
Copyright(c) 2016 Facebook Inc.

All contributions by Google :
Copyright(c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia :
Copyright(c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain :
Copyright 2019 - 2020 Kakao Brain

All contributions by Cruise LLC :
Copyright(c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe :
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions :
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe : each contributor holds
copyright over their contributions to Caffe2.The project versioning records
all such contribution and copyright details.If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met :

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and /or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
and IDIAP Research Institute nor the names of its contributors may be
used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

const std::string reduction_template =  
        R"ESC(${program_name}

#include <limits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#define INDEX_T ${indexT}
#define MAX_THREADS ${max_threads_lb}
#define INPUT_T ${inT}
#define OUTPUT_T ${outT}
#define OVT ${output_vec_size}
#define VT0 ${vt0}

${preamble}

namespace megu {

namespace memory{
// aligned vector generates vectorized load/store on CUDA
  template<typename scalar_t, int vec_size>
  struct alignas(sizeof(scalar_t) * vec_size) vector {
    scalar_t val[vec_size];
  };

}//end memory


template<typename scalar_t, int vec_size>
using gpu_vec = memory::vector<scalar_t,vec_size>;

template <int vec_size, typename scalar_t ,typename index_t = unsigned int>
     inline gpu_vec<scalar_t, vec_size> vload(const scalar_t* base_ptr, index_t offset) {
        using vec_t = gpu_vec<scalar_t, vec_size>;
        auto* from = reinterpret_cast<const vec_t*>(base_ptr);
        return from[offset];
    }

    template <int vec_size , typename index_t = unsigned int> 
     inline gpu_vec<bool, vec_size> vload(const bool* base_ptr, index_t offset) {
        auto tmp = vload<vec_size,uint8_t , index_t>(reinterpret_cast<const uint8_t*>(base_ptr), offset); 
        gpu_vec<bool, vec_size> ret;
        for (int i = 0; i < vec_size; ++i) {
            ret.val[i] = bool(tmp.val[i]);
        }
        return ret;
    }

    template<typename T>
    static  __forceinline__ T WARP_SHFL_DOWN(T value,
        unsigned int delta, int width = warpSize,
        unsigned int mask = 0xffffffff)
    {
#if !defined(__HIP_DEVICE_COMPILE__)
        return __shfl_down_sync(mask, value, delta, width);
#else
        return __shfl_down(value, delta, width);
#endif
    }

#if defined(__HIP_DEVICE_COMPILE__)
    template<>
    static  __forceinline__ int64_t WARP_SHFL_DOWN<int64_t>(int64_t value, unsigned int delta, int width, unsigned int mask)
    {
        //(HIP doesn't support int64_t). Trick from https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
        int2 a = *reinterpret_cast<int2*>(&value);
        a.x = __shfl_down(a.x, delta);
        a.y = __shfl_down(a.y, delta);
        return *reinterpret_cast<int64_t*>(&a);
    }
#endif

#if defined(MEGU_FP16_H)
  using half = float16;
  
#endif

#if defined(MEGU_COMPLEX_H)
    template <typename T>
    static __device__ __forceinline__ thrust::complex<T> WARP_SHFL_DOWN(thrust::complex<T> value,
        unsigned int delta, int width = warpSize,
        unsigned int mask = 0xffffffff)
    {
#if !defined(__HIP_DEVICE_COMPILE__)
        return thrust::complex<T>(
            __shfl_down_sync(mask, value.real(), delta, width),
            __shfl_down_sync(mask, value.imag(), delta, width));
#else
        return thrust::complex<T>(
            __shfl_down(value.real(), delta, width),
            __shfl_down(value.imag(), delta, width));
#endif
    }
#endif
    
	 static void reduce_fraction(size_t& numerator, size_t& denominator) {
		// get GCD of num and denom using Euclid's algorithm.
		// Can replace this with std::gcd if we ever support c++17.
        size_t a = denominator;
        size_t b = numerator;
        while (b != 0) {
            a %= b;
            // swap(a,b)
            size_t tmp = a;
            a = b;
            b = tmp;
        }

		// a is now the GCD
		numerator /= a;
		denominator /= a;
	}



struct ReduceConfig {
  //has to match host-side ReduceConfig in the eager code
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int input_vec_size = 4;
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

   bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

   bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

   bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

   bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

   bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

   int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
   int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size;
  }

   int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

   int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

};

//just use assert() ? it works on my machine huh
#define CUDA_KERNEL_ASSERT(...) ;

template<typename T, typename T2>
struct Pair{
   T first;
   T2 second;
};

template <typename scalar_t, typename index_t,
 typename out_scalar_t = scalar_t, int vt0 = 4>
    struct ReduceOp {

        ${impl}

        using InputCalculator = detail::IndexToOffset<1, index_t>; 
        using OutputCalculator = detail::IndexToOffset<2, index_t>;  

        static constexpr bool can_accumulate_in_output =
            std::is_convertible<arg_t, out_scalar_t>::value 
            && std::is_convertible<out_scalar_t, arg_t>::value; 

        static constexpr int input_vec_size = ReduceConfig::input_vec_size;

        ReduceConfig config;
        InputCalculator input_calc;
        OutputCalculator output_calc;
        const void* src;
        const char* dst[2]; //it accepts at most two destinations
        // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
        // output is not permissible
        void* acc_buf;
        // cta_buf used for accumulation between blocks during global reduction
        void* cta_buf;
        int* semaphores;
        int64_t base_idx;
        bool accumulate;
        bool final_output;
        int noutputs;

        ${scalars}

        template <int output_vec_size>
         void run() const {
            extern __shared__ char shared_memory[];
            index_t output_idx = config.output_idx<output_vec_size>();
            index_t input_idx = config.input_idx();
            auto base_offsets1 = output_calc.get(output_idx)[1];

            using arg_vec_t = detail::Carray<arg_t, output_vec_size>; 
            arg_vec_t value;

            if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
                const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);
                value = thread_reduce<output_vec_size>(input_slice); 
            }

            if (config.should_block_y_reduce()) {
                value = block_y_reduce<output_vec_size>(value, shared_memory);
            }
            if (config.should_block_x_reduce()) {
                value = block_x_reduce<output_vec_size>(value, shared_memory);
            }

            using out_ptr_vec_t = detail::Carray<out_scalar_t*, output_vec_size>;
            using offset_vec_t = detail::Carray<index_t, output_vec_size>;
            offset_vec_t base_offsets;
            out_ptr_vec_t out;

#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
                base_offsets[i] = output_calc.get(output_idx + i)[0];
                out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
            }

            arg_vec_t* acc = nullptr;
            if (acc_buf != nullptr) {
                size_t numerator = sizeof(arg_t);
                size_t denominator = sizeof(out_scalar_t);
                reduce_fraction(numerator, denominator);
                acc = (arg_vec_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
            }

            if (config.should_global_reduce()) {
                value = global_reduce<output_vec_size>(value, acc, shared_memory);
            }
            else if (config.should_store(output_idx)) {
                if (accumulate) {
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = translate_idx(value[i], base_idx);
                    }
                }

                if (acc == nullptr) {
                    if (accumulate) {
                        value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
                    }
                    if (final_output) {
                        set_results_to_output<output_vec_size>(value, base_offsets);
                    }
                    else {
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
                        }
                    }
                }
                else {
                    if (accumulate) {
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = combine((*acc)[i], value[i]);
                        }
                    }
                    if (final_output) {
                        set_results_to_output<output_vec_size>(value, base_offsets);
                    }
                    else {
                        *acc = value;
                    }
                }
            }
        }

        template <int output_vec_size>
         detail::Carray<arg_t, output_vec_size> thread_reduce(const scalar_t* data) const {
            if (config.vectorize_input) {
                CUDA_KERNEL_ASSERT(output_vec_size == 1);
                // reduce at the header of input_slice where memory is not aligned,
                // so that thread_reduce will have an aligned memory to work on.
                return { input_vectorized_thread_reduce_impl(data) };
            }
            else {
                index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
                bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
                if (is_contiguous) {
                    return thread_reduce_impl<output_vec_size>(data, [](index_t idx) { return idx; });
                }
                else if (input_calc.dims == 1) {
                    return thread_reduce_impl<output_vec_size>(data, [&](index_t idx) { return idx * element_stride; });
                }
                else {
                    return thread_reduce_impl<output_vec_size>(data, [&](index_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
                }
            }
        }

         arg_t input_vectorized_thread_reduce_impl(const scalar_t* data) const {
            index_t end = config.num_inputs;

            // Handle the head of input slice where data is not aligned
            arg_t value = ident;
            constexpr int align_bytes = alignof(memory::vector<scalar_t, input_vec_size>);
            constexpr int align_elements = align_bytes / sizeof(scalar_t);
            int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);
            if (shift > 0) {
                data -= shift;
                end += shift;
                if (threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()) {
                    value = reduce(value, data[threadIdx.x], threadIdx.x - shift);
                }
                end -= align_elements;
                data += align_elements;
                shift = align_elements - shift;
            }

            // Do the vectorized reduction
            using load_t = memory::vector<scalar_t, input_vec_size>;

            index_t idx = config.input_idx();
            const index_t stride = config.step_input;

            // Multiple accumulators to remove dependency between unrolled loops.
            arg_t value_list[input_vec_size];
            value_list[0] = value;

#pragma unroll
            for (int i = 1; i < input_vec_size; i++) {
                value_list[i] = ident;
            }

            while (idx * input_vec_size + input_vec_size - 1 < end) {
                const auto values_vec = vload<input_vec_size>(data, idx);
#pragma unroll
                for (index_t i = 0; i < input_vec_size; i++) {
                    value_list[i] = reduce(value_list[i], values_vec.val[i], shift + idx * input_vec_size + i);
                }
                idx += stride;
            }

            // tail
            index_t tail_start = end - end % input_vec_size;
            if (config.should_reduce_tail()) {
                int idx = tail_start + threadIdx.x;
                if (idx < end) {
                    const auto value = data[idx];
                    value_list[0] = reduce(value_list[0], value, idx + shift);
                }
            }
            
            // combine accumulators
#pragma unroll
            for (int i = 1; i < input_vec_size; i++) {
                value_list[0] = combine(value_list[0], value_list[i]);
            }
            return value_list[0];
        }

)ESC"

        
                            + 


    (std::string const)
            R"ESC(
        template <int output_vec_size, typename offset_calc_t>
         detail::Carray<arg_t, output_vec_size> thread_reduce_impl(const scalar_t* data_, offset_calc_t calc) const {
            index_t idx = config.input_idx();
            const index_t end = config.num_inputs;
            const index_t stride = config.step_input;

            using arg_vec_t = detail::Carray<arg_t, output_vec_size>;
            using load_t = memory::vector<scalar_t, output_vec_size>; 

            // Multiple accumulators to remove dependency between unrolled loops.
            arg_vec_t value_list[vt0];

#pragma unroll
            for (int i = 0; i < vt0; i++) {
#pragma unroll
                for (int j = 0; j < output_vec_size; j++) {
                    value_list[i][j] = ident;
                }
            }

            load_t values[vt0];

            while (idx + (vt0 - 1) * stride < end) {
#pragma unroll
                for (index_t i = 0; i < vt0; i++) {
                    const auto offset = calc(idx + i * stride) / output_vec_size;
                    values[i] = vload<output_vec_size>(data_, offset); 
                }
#pragma unroll
                for (index_t i = 0; i < vt0; i++) {
#pragma unroll
                    for (index_t j = 0; j < output_vec_size; j++) {
                        value_list[i][j] = reduce(value_list[i][j], values[i].val[j], idx + i * stride);
                    }
                }
                idx += stride * vt0;
            }

            // tail
            int idx_ = idx;
#pragma unroll
            for (index_t i = 0; i < vt0; i++) {
                if (idx >= end) {
                    break;
                }
                const auto offset = calc(idx) / output_vec_size;
                values[i] = vload<output_vec_size>(data_, offset);
                idx += stride;
            }
            idx = idx_;
#pragma unroll
            for (index_t i = 0; i < vt0; i++) {
                if (idx >= end) {
                    break;
                }
#pragma unroll
                for (index_t j = 0; j < output_vec_size; j++) {
                    value_list[i][j] = reduce(value_list[i][j], values[i].val[j], idx);
                }
                idx += stride;
            }

            // combine accumulators
#pragma unroll
            for (int i = 1; i < vt0; i++) {
#pragma unroll
                for (index_t j = 0; j < output_vec_size; j++) {
                    value_list[0][j] = combine(value_list[0][j], value_list[i][j]);
                }
            }
            return value_list[0];
        }

        template <int output_vec_size>
         detail::Carray<arg_t, output_vec_size> block_x_reduce(detail::Carray<arg_t, output_vec_size> value, char* shared_memory) const {
            using args_vec_t = detail::Carray<arg_t, output_vec_size>;
            int dim_x = blockDim.x;
            args_vec_t* shared = (args_vec_t*)shared_memory;
            if (dim_x > warpSize) {
                int address_base = threadIdx.x + threadIdx.y * blockDim.x;
                shared[address_base] = value;
                for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
                    __syncthreads();
                    if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                        args_vec_t other = shared[address_base + offset];
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = combine(value[i], other[i]);
                        }
                        shared[address_base] = value;
                    }
                }
                dim_x = warpSize;
            }

            __syncthreads();

            for (int offset = 1; offset < dim_x; offset <<= 1) {
#pragma unroll
                for (int i = 0; i < output_vec_size; i++) {
                    arg_t other = warp_shfl_down(value[i], offset);
                    value[i] = combine(value[i], other);
                }
            }
            return value;
        }

        template <int output_vec_size>
         detail::Carray<arg_t, output_vec_size> block_y_reduce(detail::Carray<arg_t, output_vec_size> value, char* shared_memory) const {
            using args_vec_t = detail::Carray<arg_t, output_vec_size>;
            args_vec_t* shared = (args_vec_t*)shared_memory;
            shared[config.shared_memory_offset(0)] = value;
            for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
                __syncthreads();
                if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                    args_vec_t other = shared[config.shared_memory_offset(offset)];
#pragma unroll
                    for (int i = 0; i < output_vec_size; i++) {
                        value[i] = combine(value[i], other[i]);
                    }
                    shared[config.shared_memory_offset(0)] = value;
                }
            }
            return value;
        }

         bool mark_block_finished() const {
            __shared__ bool is_last_block_done_shared;

            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
                is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
            }

            __syncthreads();

            return is_last_block_done_shared;
        }

        template <int output_vec_size, bool can_acc>
         detail::Carray<arg_t, output_vec_size> accumulate_in_output(
            detail::Carray<out_scalar_t*, output_vec_size> out,
            detail::Carray<arg_t, output_vec_size> value,
            typename std::enable_if<can_acc>::type* = nullptr
        ) const {
            detail::Carray<arg_t, output_vec_size> ret;
#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
                ret[i] = combine(*(out[i]), value[i]);
            }
            return ret;
        }

        template <bool can_acc>
         out_scalar_t get_accumulated_output(
            out_scalar_t* out, arg_t value,
            typename std::enable_if<can_acc>::type* = nullptr
        ) const {
            CUDA_KERNEL_ASSERT(!final_output); 
            return (out_scalar_t)value;
        }

        // This function should never be called --
        // it's the version of `accumulate_in_output`
        // when accumulation in the output is not possible.
        template <int output_vec_size, bool can_acc>
         detail::Carray<arg_t, output_vec_size> accumulate_in_output(
            detail::Carray<out_scalar_t*, output_vec_size>,
            detail::Carray<arg_t, output_vec_size>,
            typename std::enable_if<!can_acc>::type* = nullptr
        ) const {
            CUDA_KERNEL_ASSERT(false);
            return {};
        }

        // This function should never be called --
        // it's the version of `get_accumulated_output`
        // when accumulation in the output is not possible.
        template <bool can_acc>
         out_scalar_t get_accumulated_output(
            out_scalar_t* out, arg_t value,
            typename std::enable_if<!can_acc>::type* = nullptr
        ) const {
            CUDA_KERNEL_ASSERT(false);
            return *out;
        }

        template<class T>
         void set_results(const T x, const index_t base_offset) const {
            CUDA_KERNEL_ASSERT(noutputs == 1);
            auto res = (out_scalar_t*)((char*)dst[0] + base_offset);
            *res = x;
        }

        //Currently implemented for max of two outputs
        template<typename T1 , typename T2>
         void set_results(const Pair<T1, T2>& x, const index_t base_offset) const { 
            if (noutputs >= 1) {
                auto res0 = (T1*)((char*)dst[0] + base_offset);
                *res0 = x.first;
            }
            if (noutputs >= 2) {
                // base offset is computed assuming element size being sizeof(T1), so we need to make a
                // correction to obtain the correct base offset
                auto res1 = (T2*)((char*)dst[1] + base_offset / sizeof(T1) * sizeof(T2));
                *res1 = x.second;
            }
        }

        template <int output_vec_size>
         void set_results_to_output(detail::Carray<arg_t, output_vec_size> value, detail::Carray<index_t, output_vec_size> base_offset) const {
            CUDA_KERNEL_ASSERT(final_output);
#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
                set_results(project(value[i]), base_offset[i]);
            }
        }

        template <int output_vec_size>
         detail::Carray<arg_t, output_vec_size> global_reduce(detail::Carray<arg_t, output_vec_size> value, detail::Carray<arg_t, output_vec_size>* acc, char* shared_memory) const {
            using arg_vec_t = detail::Carray<arg_t, output_vec_size>;
            using out_ptr_vec_t = detail::Carray<out_scalar_t*, output_vec_size>;
            using offset_vec_t = detail::Carray<index_t, output_vec_size>;

            arg_vec_t* reduce_buffer = (arg_vec_t*)cta_buf;
            index_t output_idx = config.output_idx<output_vec_size>();
            offset_vec_t base_offsets;
            out_ptr_vec_t out;

#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
                base_offsets[i] = output_calc.get(output_idx + i)[0];
                out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
            }

            bool should_store = config.should_store(output_idx);
            if (should_store) {
                index_t offset = config.staging_memory_offset(blockIdx.y);
                reduce_buffer[offset] = value;
            }

            __threadfence(); // make sure writes are globally visible
            __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
            bool is_last_block_done = mark_block_finished();

            if (is_last_block_done) {
                value.fill(ident);
                if (config.should_block_x_reduce()) {
                    index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
                    index_t step = blockDim.x * blockDim.y;
                    for (; input_offset < config.ctas_per_output; input_offset += step) {
                        index_t idx = config.staging_memory_offset(input_offset);
                        arg_vec_t next = reduce_buffer[idx];
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = combine(value[i], next[i]);
                        }
                    }
                }
                else {
                    index_t input_offset = threadIdx.y;
                    index_t step = blockDim.y;
                    for (; input_offset < config.ctas_per_output; input_offset += step) {
                        index_t idx = config.staging_memory_offset(input_offset);
                        arg_vec_t next = reduce_buffer[idx];
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = combine(value[i], next[i]);
                        }
                    }
                }
                value = block_y_reduce(value, shared_memory);
                if (config.should_block_x_reduce()) {
                    value = block_x_reduce<output_vec_size>(value, shared_memory);
                }
                if (should_store) {
                    if (accumulate) {
#pragma unroll
                        for (int i = 0; i < output_vec_size; i++) {
                            value[i] = translate_idx(value[i], base_idx);
                        }
                    }

                    if (acc == nullptr) {
                        if (accumulate) {
                            value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
                        }
                        if (final_output) {
                            set_results_to_output<output_vec_size>(value, base_offsets);
                        }
                        else {
#pragma unroll
                            for (int i = 0; i < output_vec_size; i++) {
                                *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
                            }
                        }
                    }
                    else {
                        if (accumulate) {
#pragma unroll
                            for (int i = 0; i < output_vec_size; i++) {
                                value[i] = combine((*acc)[i], value[i]);
                            }
                        }
                        if (final_output) {
                            set_results_to_output<output_vec_size>(value, base_offsets);
                        }
                        else {
                            *acc = value;
                        }
                    }
                }
            }

            return value;
        }
    };


__launch_bounds__(MAX_THREADS, 4)
__global__ void reduction_${kernel_name}_kernel(ReduceOp<INPUT_T , INDEX_T ,OUTPUT_T , VT0 > r){
  r. template run< OVT >();
}

}//end megu
    )ESC";
