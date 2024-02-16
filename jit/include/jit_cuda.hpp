#pragma once
#include <bitset>
#include "types.h"
#include "../common/array_ref.hpp"
#include "jit_function_arg.hpp"
#include "../common/export.h"
#include <cuda_runtime_api.h>

namespace megu::cuda::jit
{

	std::string getCudaPath();

	struct Dim3 {
		unsigned int x = 1;
		unsigned int y = 1;
		unsigned int z = 1;
	};

	MEGU_API void launchRawKernel(uintptr_t _kernel, Dim3 const& grid, Dim3 const& block, unsigned int smem,
		uintptr_t stream, void* args[]);

	struct MEGU_API JitFunction
	{
	private:
		struct JitFunctionContext; 
	public:

		JitFunction(std::string name, std::string args, std::string code);

		void operator() (JitFunctionArg&,cudaStream_t stream =0 , bool try_vectorize = false) const;

		
		ArrayRef<std::string> function_args()const { return args_; }

		std::string_view function_name()const { return name_; }

		std::string_view function_body()const { return body_; }

	private:
		std::string body_;
		std::string name_;
		std::vector<std::string> args_;

	};

	struct MEGU_API ContentArgs
	{
		ContentArgs(const JitFunctionArg& arg , bool vectorize)
			:
			is_contiguous{ arg.is_contiguous() }, 
			can_use_32bit_indexing{ arg.can_use_32bit_indexing() }, 
			vectorization{ vectorize ? getVectorizationInfo(arg) : vec_ctx{-1,false} } 
		{}

		
		struct vec_ctx{ 
			const int vec_size;
			const bool vectorized : 1;
		}const vectorization;

		const bool is_contiguous : 1;
		const bool can_use_32bit_indexing : 1;


		static vec_ctx getVectorizationInfo(JitFunctionArg const& arg);

	};
	


}