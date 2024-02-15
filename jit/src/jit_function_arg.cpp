#include "../include/jit_function_arg.hpp"

namespace megu::cuda::jit {

	std::string_view JitFunctionArg::get_C_name(const dtype_t type)
	{
		switch (type)
		{
			using namespace std::string_view_literals;
		case megu::Bool:
			return "bool"sv;
		case megu::Int8:
			return "int8_t"sv;
		case megu::Int32:
			return "int32_t"sv;
		case megu::Int64:
			return "int64_t"sv;
		case megu::Half:
			return "megu::float16"sv;
		case megu::BFloat16:
			return "megu::bfloat16"sv;
		case megu::Float:
			return "float"sv;
		case megu::Double:
			return "double"sv;
		case megu::Complex64:
			return "thrust::complex<float>"sv;
		case megu::Complex128:
			return "thrust::complex<double>"sv;
		default:
			MEGU_ENSURE(false, "Undefined type");
		}
	}

	std::string JitFunctionArg::encode(int reserve_extra_size )const {
		std::string ss;
		const int nargs = arg.data.size() + scalars.size();
		ss.reserve(reserve_extra_size + (nargs * 4));
		int si = 0, ti = 0;
		for (int i = 0; i < nargs; ++i)
		{
			ss.push_back(char(is_const[i]));
			ss.push_back(char(is_pointer[i]));
			ss.push_back(char(is_scalar[i] ? scalars[si++].dtype() : arg.dtype[ti++]));
			ss.push_back(char(is_scalar[i]));
		}
		//ss.push_back(char(is_contiguous()));
		//ss.push_back(char(can_use_32bit_indexing()));
		return ss;
	}

	size_t JitFunctionArg::len() const {
		size_t  out = 1;
		for (int i = 0; i < arg.shape.size(); ++i)
			out *= arg.shape[i];
		return out;
	}

}//end megu::cuda::jit