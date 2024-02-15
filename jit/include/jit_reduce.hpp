#pragma once
#include "redux_arg.hpp"
#include "scalar.hpp"
#include "../common/export.h"
#include <cuda_runtime_api.h>

namespace megu {
    struct InjectedAllocator;
}

namespace megu::detail {
    struct ReduceConfig;
}
namespace megu::cuda::jit {
    struct JitScalarArg;
}

namespace megu::jit {

    //util function to get a config with arg_t == acc_type_t<scalar_t>
    MEGU_API detail::ReduceConfig getSimpleConfig(ReduxArg const& arg, const int vt0 = 4);

    MEGU_API void reductionEX(ReduxArg const& arg,
       /*
       * full impl containing the functions for project - combine - translate_idx - reduce
       * - identity def(or constexpr static) - warp_shfl_down and the arg_t typedef
       */
        std::string_view impl,
        std::string_view kernel_name, 
        detail::ReduceConfig const& config, 
        InjectedAllocator const&,
        cudaStream_t =0,
        const int vt0 = 4);

    MEGU_API  void simpleReduction(
        const ReduxArg& arg,
        std::string_view combine,// eg. "arg_t combine(arg_t x , arg_t y) {return x+y;}"
        std::string_view kernel_name,
        InjectedAllocator const& alloc,
        std::string_view identity = "(arg_t)0",
        cudaStream_t stream =0,
        const int vt0 = 4 //unrolling factor
    );


    MEGU_API void reductionEXWithScalar(
        const ReduxArg& arg,
        /*
       * full impl containing the functions for project - combine - translate_idx - reduce
       * - identity def(or constexpr static) - warp_shfl_down and the arg_t typedef
       */
        std::string_view impl,
        std::string_view kernel_name,
        const detail::ReduceConfig& config,
        const cuda::jit::JitScalarArg& extra, 
        InjectedAllocator const&,
        cudaStream_t = 0,
        const int vt0 = 4
    );

}//end megu::jit