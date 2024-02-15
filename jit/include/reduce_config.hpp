//the reduction jit works similarly to pytorch with some added features

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
#pragma once
#include "types.h" 
#include "redux_arg.hpp"
#include "jit_cuda.hpp"


namespace megu::detail {

    constexpr int last_pow2(int n) {
        n |= (n >> 1);
        n |= (n >> 2);
        n |= (n >> 4);
        n |= (n >> 8);
        n |= (n >> 16);
        return std::max(1, n - (n >> 1));
    }
    constexpr int64_t div_up(int64_t a, int64_t b) {
        return (a + b - 1) / b;
    }

    constexpr int max_reduce_threads(dtype_t type) {
        return type == Complex128 ? 256 : 512;
    }
    template <typename scalar_t>
    int get_output_vec_size(const ReduxArg& arg) {
        int vec_size = 4;
        auto update_vec_size = [&vec_size](uint64_t n) {
            while (n % vec_size != 0) {
                vec_size /= 2;
            }
        };

        uint64_t base_address = reinterpret_cast<uint64_t>(arg.data[arg.nargs() - 1]) / sizeof(scalar_t);
        update_vec_size(base_address);

        const int output_index = arg.num_reduce_dims();
        update_vec_size(arg.shape[output_index]);

        int j = 0;
        for (auto i : arg.stride[arg.nargs() - 1]) {
            if (j != output_index) {
                update_vec_size(i / sizeof(scalar_t));
            }
            j++;
        }
        return vec_size;
    }

    static inline int warpSize() {
        int dev;
        MEGU_CUDA_CHECK(cudaGetDevice(&dev));
        int warp;
        MEGU_CUDA_CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, dev));
        return warp;
    }

    static inline int threadsPerMP() {
        int dev;
        MEGU_CUDA_CHECK(cudaGetDevice(&dev));
        int t;
        MEGU_CUDA_CHECK(cudaDeviceGetAttribute(&t, cudaDevAttrMaxThreadsPerMultiProcessor, dev)); 
        return t;
    }

    static inline int MPCount() {
        int dev;
        MEGU_CUDA_CHECK(cudaGetDevice(&dev));
        int t;
        MEGU_CUDA_CHECK(cudaDeviceGetAttribute(&t, cudaDevAttrMultiProcessorCount, dev));
        return t;
    }


    struct MEGU_API ReduceConfig {
        static constexpr int BLOCK_X = 0;
        static constexpr int BLOCK_Y = 1;
        static constexpr int CTA = 2;

        static constexpr int input_vec_size = 4;

        ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
            : element_size_bytes(element_size_bytes)
            , num_inputs(num_inputs)
            , num_outputs(num_outputs) {}
        int element_size_bytes;
        int num_inputs;
        int num_outputs;
        int step_input = 1;
        int step_output = 1;
        int ctas_per_output = 1;
        int input_mult[3] = { 0, 0, 0 };
        int output_mult[2] = { 0, 0 };

        int block_width;
        int block_height;
        int num_threads;

        bool vectorize_input = false;
        int output_vec_size = 1;

        template <typename T>
        void set_block_dimension(int64_t dim0, int64_t dim1) {
            const int max_num_threads = max_reduce_threads(megu::detail::primitive_to_dtype<T>::value) / output_vec_size;
            int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0)) : max_num_threads;
            int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(last_pow2(dim1)) : max_num_threads;
            block_width = std::min(dim0_pow2, warpSize() );
            block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
            block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
            num_threads = block_width * block_height;
        }


        int split_input(int parallelism) {
            int step = step_input;
            step_input *= parallelism;
            return step;
        }

        int split_output(int parallelism) {
            int step = step_output;
            step_output *= parallelism;
            return step;
        }

        cuda::jit::Dim3 block() const {
            return cuda::jit::Dim3(block_width, block_height);
        }

        cuda::jit::Dim3 grid() const {
            return cuda::jit::Dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
        }

        bool should_block_x_reduce() const {
            return input_mult[BLOCK_X] != 0;
        }

        bool should_block_y_reduce() const {
            return input_mult[BLOCK_Y] != 0;
        }

        bool should_global_reduce() const {
            return input_mult[CTA] != 0;
        }

        int shared_memory_size() const {
            if (!should_block_y_reduce() &&
                (!should_block_x_reduce() ||
                    block_width <= warpSize())) {
                return 0;
            }
            return element_size_bytes * num_threads * output_vec_size;
        }

        int64_t global_memory_size() const {
            if (!should_global_reduce()) {
                return 0;
            }
            auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
            if (!should_block_x_reduce()) {
                size *= block().x * output_vec_size;
            }
            return size;
        }

        int semaphore_size() const {
            if (!should_global_reduce()) {
                return 0;
            }
            return sizeof(int) * grid().x;
        }

        int values_per_thread() const {
            return div_up(num_inputs, step_input);
        }
    };

    template<typename arg_t, typename scalar_t>
    ReduceConfig setReduceConfig(const ReduxArg& arg, int vt0) {
        // Start by assuming that each thread handles a single output and all
        // the inputs for that output.
        int64_t num_outputs = arg.output_len();
        int64_t inputs_per_output = arg.len() / num_outputs;
        int input_index = arg.nargs() - 1;

        auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

        int64_t dim0;
        int64_t dim1;
        int64_t fastest_moving_stride;
        bool reduction_on_fastest_striding_dimension;

        if (arg.ndim() > 0) {
            // Adjust block size to map block width to fastest changing dimension of input
            // tensor. This grants the best possible memory accessing pattern, given that
            // for non-contiguous tensor with space in between, we cannot have perfect
            // memory coalescing.
            reduction_on_fastest_striding_dimension =
                (arg.num_reduce_dims() == arg.ndim()) ||
                (arg.stride[/*arg=*/input_index][0] <
                    arg.stride[/*arg=*/input_index][arg.num_reduce_dims()]);
            // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
            // dim0 & dim1 are more like the upper bound of the block dimension. The
            // actual launch config and reduction scheme is determined by setting values
            // to `config.input_mult` and `config.output_mult`.
            // We try to max out dim1 so that we have enough threads per CTA to deliver
            // performance for larger problem size.
            if (reduction_on_fastest_striding_dimension) {
                // Map block.x to the fastest reducing dimension. It implies:
                //   1. block_x_reduce is required.
                //   2. block.y now max out to num_outputs.
                dim0 = inputs_per_output;
                dim1 = num_outputs;
                fastest_moving_stride = arg.stride[/*arg=*/input_index][0];
            }
            else {
                // Map block.x to the fastest non reducing dimension. It implies:
                //   1. block_x_reduce is turned off.
                //   2. block.y now max out to inputs_per_output.
                dim0 = num_outputs;
                dim1 = inputs_per_output;
                fastest_moving_stride = arg.stride[/*arg=*/input_index][arg.num_reduce_dims()];
            }
        }
        else {
            reduction_on_fastest_striding_dimension = true;
            fastest_moving_stride = sizeof(scalar_t);
            dim0 = 1;
            dim1 = 1;
        }

        // We do vectorization to gain better memory access, there are two cases which we call
        // "vectorize along input" and "vectorize along output". Note that the "input/output"
        // here does not mean we are vectorizing load/store instructions. We always only vectorize
        // load instructions.
        //
        // Case 1: "vectorize along input"
        // This case happens when we are reducing along fastest moving dimesion. In such case, threads
        // with the same threadIdx.y works on the same reduction cooperatively and will produce results
        // for the same ouput. In such case, values in each loaded vector always correspond to the same ouput.
        //
        // Case 2: "vectorize along output"
        // This case happens when the fastest moving dimesion is not the dimension of reduction. In such case,
        // threads with different threadIdx.x are independent and will produce results for different outputs.
        // In such case, values in each loaded vector always correspond to different outputs.
        if (fastest_moving_stride == sizeof(scalar_t)) {
            if (reduction_on_fastest_striding_dimension && dim0 > 128 && arg.num_reduce_dims() == 1 
                && vt0 >= ReduceConfig::input_vec_size) {
                // Case 1: "vectorize along input"
                // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
                // we should avoid vectorization.
                config.vectorize_input = true;
                dim0 /= config.input_vec_size;
            }
            else if (!reduction_on_fastest_striding_dimension) {
                // Case 2: "vectorize along output"
                config.output_vec_size = get_output_vec_size<scalar_t>(arg);
                dim0 /= config.output_vec_size;
            }
        }

        // Adjust block_width and block_height
        config.set_block_dimension<scalar_t>(dim0, dim1);

        int block_width = config.block_width;
        int block_height = config.block_height;

        if (arg.ndim() == 0 || reduction_on_fastest_striding_dimension) {
            // Split the input across lanes if the input is contiguous in the reduced
            // dimension. This will require reduction between threads using warp
            // shuffle instructions and shared memory (if block_width > warpSize).
            config.input_mult[0] = config.split_input(block_width);
        }
        else {
            // Otherwise split the output across lanes in a warp.
            config.output_mult[0] = config.split_output(block_width);
        }

        constexpr int min_values_per_thread = 16;
        constexpr int max_values_per_thread = 256;

        if (config.values_per_thread() >= block_height * 16 || config.values_per_thread() >= max_values_per_thread) {
            // Divide the input across warps in a thread-block, if that leaves at least
            // 16 elements to be summed by each thread. This will require inter-warp
            // reduction using shared memory.
            config.input_mult[1] = config.split_input(block_height);
        }
        else {
            // Otherwise, each warp handles a separate output.
            config.output_mult[1] = config.split_output(block_height);
        }

        const int blocks_per_sm = threadsPerMP() / config.num_threads;
            const int num_mp = MPCount();
        const int target_grid_size = num_mp * blocks_per_sm;
        int grid = config.grid().x;
        if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
            // Divide the input across thread-blocks if the amount of work per-thread
            // is large enough and the size of the output is small enough. This will
            // require a reduction using global memory.
            // If we decide to split input across blocks, as long as we can get enough
            // number of blocks (`target_grid_size`) to balance SM, we should still
            // make the number of values per thread large for best performance.
            int ctas_per_output1 = div_up(target_grid_size, grid);
            int ctas_per_output2 = div_up(config.values_per_thread(), min_values_per_thread);
            int ctas_per_output3 = div_up(config.values_per_thread(), max_values_per_thread);
            // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
            // a large number of values to deal with. But we don't want values_per_thread to be larger than
            // max_values_per_thread
            config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
            if (config.ctas_per_output > 1) {
                config.input_mult[2] = config.split_input(config.ctas_per_output);
            }
        }
        return config;
    };
}//edn megu::detail