#pragma once
#include "alloc_inject.hpp"
#include "tensor_morphology.hpp"
#include "types.h"
#include "../common/export.h"
#include "../common/macros.h"
#include "../common/shape_tricks.hpp"

namespace megu {

	struct MEGU_API JitTensor {

		constexpr JitTensor() = default;

		JitTensor(uintptr_t ptr,
			std::vector<int64_t> shape,
			std::optional<std::vector<int64_t>>stride,
			Device dev,
			dtype_t dtype) 
		{
			std::vector<int64_t> stride_;
			if (stride) {
				stride_ = ArrayRef(*stride).vec();
			}
			else {
				stride_.resize(shape.size());
				memory::computeContiguousStrides(stride_.data(),shape.cbegin(),shape.cend());
			}
			void* mem = reinterpret_cast<void*>(ptr);
			impl_ = std::make_unique<JitTensorImpl>(mem, shape, stride_, dev, dtype); 
		}

		JitTensor(void* buffview, 
			LongArrayView shape, 
			LongArrayView strides,
			Device dev,
			dtype_t dtype)
			:impl_(std::make_unique<JitTensorImpl>(buffview, shape, strides , dev, dtype)) {}

		JitTensor(mem_ptr owned,
			LongArrayView shape,
			LongArrayView strides,
			Device dev,
			dtype_t dtype)
			: impl_(std::make_unique<JitTensorImpl>(std::move(owned),shape, strides, dev , dtype)) {}

		JitTensor unsafe_view(
			LongArrayView shape,
			LongArrayView strides
		)const {
			if (impl_->is_owned()) {
				return JitTensor( 
					impl_->maybe_owned_,  
					shape,
					strides,
					device(),
					dtype()
				);
			}
			return JitTensor( 
				data(),
				shape,
				strides,
				device(), 
				dtype()
			);
		}

		int64_t nbytes()const {
			return element_size() * size();
		}

		int64_t element_size()const {
			return detail::type_size(dtype());
		}

		int64_t size() const{
			return shape_tricks::prod(shape());
		}

		JitTensor broadcast_to(LongArrayView shape)const {
			return unsafe_view(
				shape,
				shape_tricks::broadcast_stride(*this, shape)
			);
		}

		JitTensor broadcast_to(JitTensor const&x)const {
			return broadcast_to(x.shape());
		}

		JitTensor clone()const{
			return JitTensor(std::make_unique<JitTensorImpl>(*impl_)); 
		}

		int64_t udim(int idx)const { return  impl_->morph_.udim(idx); }
		int64_t ustride(int idx)const { return  impl_->morph_.ustride(idx); }

		int rank()const { return impl_->morph_.ndim(); }

		LongArrayView shape()const { return  impl_->morph_.shape(); }
		LongArrayView strides()const { return  impl_->morph_.stride(); } 

		dtype_t dtype()const { return impl_->dtype_; }

		
		void * data() const{
			return impl_->buff_;
		}

		bool realized()const { return impl_ != nullptr; }

		Device device()const { return impl_->dev_; }

		
	private:
		struct JitTensorImpl {

			JitTensorImpl(void* buffview,
				LongArrayView shape,
				LongArrayView strides,
				Device dev,
				dtype_t dtype)
				:buff_(buffview), morph_(shape, strides), dtype_(dtype), dev_(dev)
			{
				MEGU_ENSURE(buffview != nullptr, "Provided memory cannot be null")
			}

			JitTensorImpl(mem_ptr owned,
				LongArrayView shape,
				LongArrayView strides,
				Device dev,
				dtype_t dtype)
				: maybe_owned_(std::move(owned)), morph_(shape, strides), dtype_(dtype), dev_(dev)
			{
				MEGU_ENSURE(maybe_owned_ != nullptr, "Provided memory cannot be null");
				buff_ = maybe_owned_.get();
			}

			bool is_owned()const {
				return maybe_owned_ != nullptr;
			}

			TensorMorphology morph_;
			mem_ptr maybe_owned_ = nullptr;
			void* buff_ = nullptr;
			Device dev_ = 0;
			dtype_t dtype_;
		};
		
		
		JitTensor(std::unique_ptr<JitTensorImpl> ptr)
			:impl_(std::move(ptr)) {}


		std::unique_ptr<JitTensorImpl> impl_ = nullptr;
	};
}//end megu