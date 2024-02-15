#pragma once
#include "../include/jit_reduce.hpp"
#include <numeric>
#include "../include/types.h"
#include "../common/cindexer.hpp"
#include "../jitify/jitify.hpp"
#undef max
#undef min
#include  "../include/codegen.hpp"
#include "../include/reduce_config.hpp"
#include "../include/redux_arg.hpp"
#include "../include/jit_scalar.hpp"
#include "../common/visit.h"
#include "../include/alloc_inject.hpp"

/*
* this is using pretty much torch's Jiterator but with some changes and added features it also uses 64bit indexing
* (i dont know how efficient that is vs splitting since im also using cutlass's int64 fast-div it is subject to change)
* 
*/
namespace megu::jit {

    
    template <typename index_t>
    static detail::IndexToOffset<2, index_t> make_output_calculator(
        const ReduxArg& arg)
    {
        int num_reduce_dims = arg.num_reduce_dims();
        int num_output_dims = arg.ndim() - num_reduce_dims;
        int input_index = arg.nargs() - 1;
        int output_index = 0;
        std::array<const int64_t*, 2> strides = {
          arg.stride[output_index].data() + num_reduce_dims,
          arg.stride[input_index].data() + num_reduce_dims,
        };
        auto shape = arg.shape.data() + num_reduce_dims;
        detail::IndexToOffset<2, index_t>out;
        out.dims = num_output_dims;
        for (auto i = 0; i < num_output_dims; ++i) {
            out.shape[i] = IntDivider<index_t>(shape[i]);
            out.strides_[i][0] = strides[0][i];
            out.strides_[i][1] = strides[1][i];
        }
        return out;
    }

    template <typename index_t>
    static detail::IndexToOffset<1, index_t> make_input_calculator(const ReduxArg& arg) {
        int num_reduce_dims = arg.num_reduce_dims();
        int input_index = arg.nargs() - 1;
        std::array<const int64_t*, 1> strides = {
          arg.stride[input_index].data(),
        };
        auto shape = arg.shape.data();
        detail::IndexToOffset<1, index_t>out;
        out.dims = num_reduce_dims;
        for (auto i = 0; i < num_reduce_dims; ++i) {
            out.shape[i] = IntDivider<index_t>(shape[i]);
            out.strides_[i][0] = strides[0][i];
        }
        return out;
    }



   


    template<typename index_t>
    struct JitReduction {

        using InputCalculator = detail::IndexToOffset<1, index_t>;
        using OutputCalculator = detail::IndexToOffset<2, index_t>;

        static constexpr bool can_accumulate_in_output =
            std::is_convertible<float, float>::value
            && std::is_convertible<float, float>::value;


        static constexpr int input_vec_size = detail::ReduceConfig::input_vec_size;

        //ops_t ops;
        //arg_t ident;

        detail::ReduceConfig config; 
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

        JitReduction(
            detail::ReduceConfig config, 
            InputCalculator input_calc,
            OutputCalculator output_calc,
            const void* src,
            char* dst0,
            char* dst1,
            void* acc_buf,
            void* cta_buf,
            int* semaphores,

            int noutputs,
            int64_t base_idx)
            :
            config(config),
            input_calc(input_calc),
            output_calc(output_calc),
            src(src),
            acc_buf(acc_buf),
            cta_buf(cta_buf),
            semaphores(semaphores),
            base_idx(base_idx),
            noutputs(noutputs) {
            dst[0] = dst0;
            if (dst1 != nullptr) {
                dst[1] = dst1;
            }
        }
    };

    template<typename Type,typename index_t>
    struct JitReductionWithScalar {
        JitReduction<index_t> op;
        Type extra;
    };

    static inline std::string getWithScalarJitCacheKey(ReduxArg const& arg,
        detail::ReduceConfig const& config,
        std::string_view impl,
        dtype_t scalar_dt, 
        const int vt0)
    {
        //encode:
        std::ostringstream ss;
        ss << "extra";
        ss << int(scalar_dt);
        ss << int(arg.dtype[0]);
        ss << int(arg.dtype[1]);
        if (arg.dtype.size() == 3) {
            ss << int(arg.dtype[2]);
        }
        ss << int(arg.compute_type);
        ss << config.output_vec_size;
        ss << vt0;
        ss << int(arg.can_use_32bit_index());
        ss << impl;
        return ss.str();
    }

    static inline std::string getSimpleJitCacheKey(ReduxArg const& arg,
        detail::ReduceConfig const& config,
        std::string_view impl ,
        const int vt0)
    {
        //encode:
        std::ostringstream ss;
        ss << int(arg.dtype[0]);
        ss << int(arg.dtype[1]);
        if (arg.dtype.size() == 3) {
            ss << int(arg.dtype[2]);
        }
        ss << int(arg.compute_type);
        ss << config.output_vec_size;
        ss << vt0;
        ss << int(arg.can_use_32bit_index());
        ss << impl; 
        return ss.str();
    }

    template<typename JitReduceOp>
    void launchJitKernel(ReduxArg const& arg,
        detail::ReduceConfig const& config,
        const int vt0,
        JitReduceOp const& op, 
        std::string_view impl,
        std::string_view kernel_name,
        std::string key, 
        std::unordered_map<std::string, jitify::experimental::KernelInstantiation>& jit_cache, 
        std::mutex& jit_mutex,
        cudaStream_t stream,
        std::optional<cuda::jit::JitScalarArg> const& opt_scalar = {} )
    {

        jitify::experimental::KernelInstantiation const* kern; 
        auto hit = jit_cache.find(key);
        if (hit == jit_cache.end()) {
            auto code = cuda::jit::codegen_reduction(
                impl,
                kernel_name,
                detail::max_reduce_threads(arg.compute_type) / config.output_vec_size, 
                vt0,
                config.output_vec_size,
                arg.can_use_32bit_index(),
                arg.dtype.back(),
                arg.dtype[0],
                arg.compute_type,
                opt_scalar
            );
            
            auto ProgramName = std::string(kernel_name) + "_program";
            const auto KernelName = std::string("megu::reduction_") + std::string(kernel_name) + "_kernel";

            std::lock_guard<std::mutex> lock(jit_mutex);

            auto item =
                jit_cache.insert({ std::move(key), 
                    jitify::experimental::Program( 
                        std::move(code),{}, 
                        {"-std=c++17" ,"--use_fast_math",
                        "-I" + cuda::jit::getCudaPath()})
                    .kernel(KernelName)
                    .instantiate() 
                    });
            kern = &item.first->second; 
            //std::cout << link->ptx();
        }
        else {
            kern = &hit->second;
        }

        auto block = config.block();
        auto grid = config.grid();

        int shared_memory = config.shared_memory_size();

        void* inargs[] = { (void*)&op };

        cuda::jit::launchRawKernel( 
            (uintptr_t)kern->operator CUfunction(), grid, block, shared_memory, (uintptr_t)stream,
            inargs
        );
    }

    detail::ReduceConfig getSimpleConfig(ReduxArg const& arg, const int vt0)
    {
        MEGU_VISIT_DTYPE(
            arg.dtype.back(), scalar_t,
            return setReduceConfig<detail::acc_type_t<scalar_t>, scalar_t>(arg, vt0);
        )
    }

    void reductionEX(ReduxArg const& arg,
        std::string_view impl,
        std::string_view kernel_name,
        detail::ReduceConfig const& config,
        InjectedAllocator const& alloc,
        cudaStream_t stream,
        const int vt0
        )
    {
        const char* in_data = (char*)arg.data[arg.nargs() - 1];
        char* out_data = (char*)arg.data[0];
        const auto noutputs = arg.nargs() - 1;
        char* out_data_extra = nullptr;
        if (noutputs > 1) {
            out_data_extra = (char*)arg.data[1];
        }

        char* acc_data = nullptr;// i dont use accumulation buffer i use 64bit index

        //ReduceConfig config = getSimpleConfig(arg, vt0);

        std::shared_ptr<void> buffer;
        std::shared_ptr<void> semaphores;
        if (config.should_global_reduce()) {
            buffer = alloc(config.global_memory_size());
            semaphores = alloc(config.semaphore_size());

            MEGU_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
        }

        static std::unordered_map<std::string, jitify::experimental::KernelInstantiation> jit_cache;
        static std::mutex jit_mutex{};

        if (arg.can_use_32bit_index()) {
            using index_t = unsigned int;
            auto output_calc = make_output_calculator<index_t>(arg);
            auto input_calc = make_input_calculator<index_t>(arg);
            auto reduce = JitReduction<index_t>(
                config,
                input_calc,
                output_calc,
                in_data,
                out_data,
                out_data_extra,
                acc_data,
                buffer.get(),
                (int*)semaphores.get(),
                noutputs,
                0/*base_idx is always zero since theres no subsequent calls and i just use 64bit index*/);

            reduce.accumulate = false;
            reduce.final_output = true;

            launchJitKernel(arg, config, vt0, reduce, std::move(impl), kernel_name, getSimpleJitCacheKey(arg,config,impl,vt0),
                jit_cache, jit_mutex,stream);
        }
        else {
            using index_t = uint64_t;
            auto output_calc = make_output_calculator<index_t>(arg);
            auto input_calc = make_input_calculator<index_t>(arg);
            auto reduce = JitReduction<index_t>(
                config,
                input_calc,
                output_calc,
                in_data,
                out_data,
                out_data_extra,
                acc_data,
                buffer.get(),
                (int*)semaphores.get(),
                noutputs,
                0/*base_idx is always zero since theres no subsequent calls and i just use 64bit index*/);

            reduce.accumulate = false;
            reduce.final_output = true;

            launchJitKernel(arg, config, vt0, reduce, std::move(impl), kernel_name, getSimpleJitCacheKey(arg, config, impl, vt0),  
                jit_cache, jit_mutex,stream);
        }
    }


    void simpleReduction(
        const ReduxArg& arg,
        std::string_view combine,// eg. "arg_t combine(arg_t x , arg_t y) {return x+y;}"
        std::string_view kernel_name,
        InjectedAllocator const& alloc,
        std::string_view identity ,
        cudaStream_t stream,
        const int vt0 
        )  
    {
        auto const impl = cuda::jit::simple_reduction_impl_code(combine, identity);
        const auto config = getSimpleConfig(arg, vt0);
        reductionEX(arg, impl, kernel_name, config, alloc,stream , vt0); 
    }


    void reductionEXWithScalar( 
        const ReduxArg& arg,
        std::string_view impl, 
        std::string_view kernel_name,
        const detail::ReduceConfig& config,
        const cuda::jit::JitScalarArg& extra, 
        InjectedAllocator const& alloc,
        cudaStream_t stream ,
        const int vt0

    ) {
        const char* in_data = (char*)arg.data[arg.nargs() - 1];
        char* out_data = (char*)arg.data[0];
        const auto noutputs = arg.nargs() - 1;
        char* out_data_extra = nullptr;
        if (noutputs > 1) {
            out_data_extra = (char*)arg.data[1];
        }

        char* acc_data = nullptr;// i dont use accumulation buffer i use 64bit index

        //ReduceConfig config = getSimpleConfig(arg, vt0);

        std::shared_ptr<void> buffer;
        std::shared_ptr<void> semaphores;
        if (config.should_global_reduce()) {
            buffer = alloc(config.global_memory_size());
            semaphores = alloc(config.semaphore_size());

            MEGU_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
        }

        static std::unordered_map<std::string, jitify::experimental::KernelInstantiation> jit_cache;
        static std::mutex jit_mutex{};

        auto key = getWithScalarJitCacheKey(arg, config, impl, extra.dtype(), vt0);
        
        MEGU_VISIT_DTYPE(extra.dtype(), scalar_t,
            [&]
            { 
                scalar_t Extra = extra.value(); 
                if (arg.can_use_32bit_index()) {
                    using index_t = unsigned int;
                    auto output_calc = make_output_calculator<index_t>(arg);
                    auto input_calc = make_input_calculator<index_t>(arg);
                    auto reduce = JitReductionWithScalar<scalar_t, index_t>{
                        JitReduction<index_t>(
                        config,
                        input_calc,
                        output_calc,
                        in_data,
                        out_data,
                        out_data_extra,
                        acc_data,
                        buffer.get(),
                        (int*)semaphores.get(),
                        noutputs,
                        0/*base_idx is always zero since theres no subsequent calls and i just use 64bit index*/),
                        Extra
                    };

                    reduce.op.accumulate = false;
                    reduce.op.final_output = true;

                    launchJitKernel(arg, config, vt0, reduce, std::move(impl), kernel_name, std::move(key),
                        jit_cache, jit_mutex , stream,extra);
                }
                else {
                    using index_t = uint64_t;
                    auto output_calc = make_output_calculator<index_t>(arg);
                    auto input_calc = make_input_calculator<index_t>(arg);
                    auto reduce = JitReductionWithScalar<scalar_t, index_t>{
                        JitReduction<index_t>(
                        config,
                        input_calc,
                        output_calc,
                        in_data,
                        out_data,
                        out_data_extra,
                        acc_data,
                        buffer.get(),
                        (int*)semaphores.get(),
                        noutputs,
                        0/*base_idx is always zero since theres no subsequent calls and i just use 64bit index*/),
                        Extra
                    };
                    reduce.op.accumulate = false;
                    reduce.op.final_output = true;

                    launchJitKernel(arg, config, vt0, reduce, std::move(impl), kernel_name, std::move(key),
                        jit_cache, jit_mutex , stream,extra);
                }
            }();
        );
    }

}//end megu