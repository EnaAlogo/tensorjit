#include "../include/redux_arg.hpp"

namespace megu {

	static inline void permuteArgs(ArrayRef<JitTensor*> args, ArrayRef<int> perms)
	{
		for (auto* arg : args) {
			arg->inplace_permute(perms);
		}

	}

	static SmallVector<int, 5> getReductionPerms(const JitTensor& broadcasted_out)
	{
		SmallVector<int, 5>ret;
		int rank = broadcasted_out.rank();
		const int64_t* strides = broadcasted_out.strides().data();
		for (int i = 0; i < rank; ++i)
		{
			if (strides[i] == 0)
				ret.push_back(i);
		}
		for (int i = 0; i < rank; ++i)
		{
			if (strides[i] != 0)
				ret.push_back(i);
		}
		return ret;
	}

	static inline bool shouldPermute(ArrayRef<int> axes)
	{
		for (int i = 0; i < axes.size(); ++i)
		{
			if (axes[i] != i)
				return true;
		}
		return false;
	}

    ReduxArg::ReduxArg(JitTensor const& input, JitTensor const& output)
	{
		MEGU_ENSURE(input.realized() && output.realized(),
			"[Reduction]All tensor args should be realized");

		auto inout = output.broadcast_to(input); 
		auto copyin = input.clone();

		const auto axes = getReductionPerms(inout);

		if (shouldPermute(axes)) {
			permuteArgs({ &inout,&copyin }, axes);
		}
		data.push_back(output.data()); 
		data.push_back(input.data()); 
		stride.emplace_back(inout.strides());
		stride.emplace_back(copyin.strides());
		dtype.push_back(output.dtype()); 
		dtype.push_back(input.dtype()); 
		compute_type = dtype[1]; 
		shape = ShapeVector(inout.shape());
		{
			auto [Shape, Keep] = shape_tricks::squash_shape(shape, stride[0], stride[1]);
			shape = std::move(Shape);
			stride[0] = shape_tricks::get_squashed_strides(stride[0], Keep);
			stride[1] = shape_tricks::get_squashed_strides(stride[1], Keep);
		}
		for (int i = 0; i < nargs(); ++i) {
			int const typesize = detail::type_size(dtype[i]);
			std::transform(stride[i].begin(), stride[i].end(),
				stride[i].begin(), [typesize](const auto& x) {return x * typesize; });
		}

    }
    ReduxArg::ReduxArg(JitTensor const& input, JitTensor const& out, JitTensor const& out2) {
		MEGU_ENSURE(input.realized() && out.realized() && out2.realized(),
			"[Reduction]All tensor args should be realized");

		MEGU_ENSURE(out.shape() == out2.shape() && out.strides() == out2.strides(), 
			"make_reduction : Both outputs should have the same shape and stride");

		auto inout = out.broadcast_to(input); 
		auto inout2 = out2.broadcast_to(input);  
		 
		auto copyin = input.clone();

		const auto axes = getReductionPerms(inout);  

		if (shouldPermute(axes)) {
			permuteArgs({ &inout,&inout2,&copyin }, axes); 
		}

		data.push_back(out.data());
		data.push_back(out2.data());
		data.push_back(input.data());
		stride.emplace_back(inout.strides());
		stride.emplace_back(inout2.strides());
		stride.emplace_back(copyin.strides());
		dtype.push_back(out.dtype());
		dtype.push_back(out2.dtype());
		dtype.push_back(input.dtype());
		compute_type = dtype[1];

		shape = ShapeVector(inout.shape());
		{
			auto [Shape, Keep] = shape_tricks::squash_shape(shape, stride[0], stride[1], stride[2]);
			shape = Shape;
			stride[0] = shape_tricks::get_squashed_strides(stride[0], Keep);
			stride[1] = shape_tricks::get_squashed_strides(stride[1], Keep);
			stride[2] = shape_tricks::get_squashed_strides(stride[2], Keep);
		}

		for (int i = 0; i < nargs(); ++i) {
			int const typesize = detail::type_size(dtype[i]);
			std::transform(stride[i].begin(), stride[i].end(),
				stride[i].begin(), [typesize](const auto& x) {return x * typesize; });
		}

    }

}