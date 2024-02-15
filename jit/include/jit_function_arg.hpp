#pragma once
#include <bitset>
#include "types.h"
#include "scalar.hpp"
#include "../common/kernel_dto.hpp"
#include "../common/carray.h"
#include "../common/export.h"

namespace megu::cuda::jit
{

	struct MEGU_API JitFunctionArg
	{
		JitFunctionArg(ElementWiseArgument const& arg , Device dev)
			:arg(arg) , device(dev){}

		static std::string_view get_C_name(const dtype_t type);  

		bool is_contiguous() const {
			return nargs() == is_pointer.count() ? true :  arg.is_contiguous();
		}

		std::string encode(int reserve_extra_size = 0)const;

		std::string_view get_typename(int i)const { return get_C_name(arg.dtype[i]); }; 
		bool can_use_32bit_indexing()const { return len() < std::numeric_limits<int>::max(); }
		bool is_input(int i)const { return is_const[i]; }
		bool is_output(int i)const { return !is_const[i]; }
		int ninputs() const { return is_const.count(); }
		int noutputs() const { return  nargs() - (ninputs() - is_scalar.count()); }
		int ndim() const { return arg.shape.size(); };
		int nargs() const { return arg.data.size(); };
		size_t len() const;

		template<int args>
		detail::Carray<void*, args> get_data_ptrs()const
		{
			detail::Carray<void*, args> out{};
			//std::memcpy(&out.data_[0], data.data(), sizeof(void*) * args);
			for (int i = 0; i < args; ++i)
				out[i] = arg.data[i];
			return out;
		}
		/*
		* fields
		*/
		//SmallVector<ShapeVector, 5> stride;
		
		//SmallVector<Scalar, 8> scalars;
		std::vector<Scalar> scalars;
		// shape;
		//SmallVector<void*,  6> data;
		//SmallVector<dtype_t,6> dtypes;
		//dtype_t compute_type;

		ElementWiseArgument const& arg;

		std::bitset<24> is_const{ 0 };
		std::bitset<24> is_pointer{ 0 };
		std::bitset<24> is_scalar{ 0 };
		Device device;
	};

	


}//end megu::cuda::jit