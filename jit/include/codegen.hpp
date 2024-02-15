#pragma once
#include "jit_cuda.hpp"
#include "jit_scalar.hpp"
#include <optional>

namespace megu::cuda::jit
{


    MEGU_API std::string codegen(
        JitFunction const& f,
        JitFunctionArg const& arg,
        ContentArgs const& details
    );

    MEGU_API std::string format_code(std::string_view code);


    MEGU_API std::string codegen_reduction(std::string_view impl,
        std::string_view kernel_name,
        int max_threads,
        int vt0,
        int output_vec_size,
        bool can_use32bit,
        dtype_t in_dtype,
        dtype_t out_type,
        dtype_t compute_type,
        std::optional<JitScalarArg> const&  scalar = {});


    MEGU_API std::string simple_reduction_impl_code(
        std::string_view const combine,
        std::optional<std::string_view> identity = std::nullopt,//defaults to 0
        std::optional<std::string_view> helper_functions = std::nullopt ,//defaults to nothing it doesnt need to exist
        std::optional<std::string_view> reduce = std::nullopt,//defaults to ignoring the index and performing the combine again
        std::optional<std::string_view> project = std::nullopt,//defaults to just casting to out_scalar_t
        std::optional<std::string_view> interm_type = std::nullopt//defaults to accum_type<T> and will be the arg_t type
    );


}//end megu::cuda::jit