#pragma once
#include "../include/memory_format.h"
#include "../include/shape_vector.hpp"
#include "../common/array_ref.hpp"

namespace megu {

    extern template class SmallVector<int64_t, 10>; 

    struct MEGU_API TensorMorphology {
        enum FromSelfView_t { FROM_SELF_VIEW };

        using MutableArrayView = MutableArrayRef<int64_t>;

        
        using TensorShapeAndStride_t = SmallVector<int64_t, 10 >;
        using Self = TensorMorphology;

        TensorMorphology() = default;

        TensorMorphology(FromSelfView_t, LongArrayView view)
            : packed_shape_and_strides_(view) {};

        TensorMorphology(LongArrayView shape) {
            packed_shape_and_strides_.resize((shape.size() << 1));
            std::memcpy(packed_shape_and_strides_.data(), shape.data(), shape.size() * sizeof(int64_t));
            memory::computeContiguousStrides(stride_data(), shape.begin(), shape.end());
        }
        TensorMorphology(LongArrayView shape, LongArrayView stride) {
            assert(shape.size() == stride.size());
            packed_shape_and_strides_.resize((shape.size() << 1));
            std::memcpy(packed_shape_and_strides_.data(), shape.data(), shape.size() * sizeof(int64_t));
            std::memcpy(packed_shape_and_strides_.data() + shape.size(), stride.data(), shape.size() * sizeof(int64_t));
        }
        TensorMorphology(LongArrayView shape , memory::memory_format_t format) {
            packed_shape_and_strides_.resize((shape.size() << 1));
            std::memcpy(packed_shape_and_strides_.data(), shape.data(), shape.size() * sizeof(int64_t));
            memory::computeStrides(stride_data(),shape_begin() , shape_begin() + shape.size() , format);
        }

         void push_back(int64_t dim, int64_t stride) {
            packed_shape_and_strides_.insert(packed_shape_and_strides_.begin() + (packed_shape_and_strides_.size() >> 1), dim);
            packed_shape_and_strides_.push_back(stride);
        }
         void pop_back() {
            packed_shape_and_strides_.erase(packed_shape_and_strides_.begin() + (packed_shape_and_strides_.size() >> 1) - 1);
            packed_shape_and_strides_.pop_back();
        }
         void insert(size_t position, int64_t dim, int64_t stride) {
            assert(position < ndim());
            packed_shape_and_strides_.insert(packed_shape_and_strides_.data() + position + ndim(), dim);
            packed_shape_and_strides_.insert(packed_shape_and_strides_.data() + (position + 1), stride);
        }
         void erase(size_t position) {
            assert(position < ndim());
            packed_shape_and_strides_.erase(packed_shape_and_strides_.data() + position);
            packed_shape_and_strides_.erase(packed_shape_and_strides_.data() + position - 1);
        }
         MutableArrayView mutable_shape() {
            return MutableArrayView(shape_begin(), shape_end());
        }
         MutableArrayView mutable_stride() {
            return MutableArrayView(stride_begin(), stride_end());
        }
         LongArrayView shape() const {
            return LongArrayView(shape_begin(), shape_end());
        }
         LongArrayView stride() const {
            return LongArrayView(stride_begin(), stride_end());
        }
         LongArrayView get_shape_strides() const {
            return packed_shape_and_strides_;
        }

         int64_t udim(size_t at) const {
            return packed_shape_and_strides_[at];
        }
         int64_t& udim(size_t at) {
            return packed_shape_and_strides_[at];
        }

         int64_t ustride(size_t at) const {
            return packed_shape_and_strides_[at + ndim()];
        }
         int64_t& ustride(size_t at) {
            return packed_shape_and_strides_[at + ndim()];
        }

         int64_t sdim(int at) const {
            return packed_shape_and_strides_[at >= 0 ? at : at + ndim()];
        }
         int64_t& sdim(int at) {
            return packed_shape_and_strides_[at >= 0 ? at : at + ndim()];
        }

         int64_t sstride(int at) const {
            int size = ndim();
            return packed_shape_and_strides_[(at >= 0 ? at : at + size) + size];
        }
         int64_t& sstride(int at) {
            int size = ndim();
            return packed_shape_and_strides_[(at >= 0 ? at : at + size) + size];
        }

        template<typename Integer>
        ShapeVector reduceShape(ArrayRef<Integer> axes) const {
            ShapeVector shape{ this->shape() };
            for (int i = 0; i < axes.size(); ++i) {
                int dim = axes[i];
                dim = dim >= 0 ? dim : dim + axes.size();
                shape[dim] = 1;
            }
            return shape;
        }

         bool equalShape(const Self& other) const {
            return shape() == other.shape();
        }
         bool equalStride(const Self& other) const {
            return stride() == other.stride();
        }
         bool fullEquality(const Self& other) const {
            return packed_shape_and_strides_ == other.packed_shape_and_strides_;
        }

        template<typename Integer>
        Self permute(ArrayRef<Integer>  perms) const {
            int size = ndim();
            Self ret;
            ret.packed_shape_and_strides_.resize(packed_shape_and_strides_.size());
            for (int i = 0; i < perms.size(); ++i) {
                int perm = perms[i];
                ret.packed_shape_and_strides_[i] = packed_shape_and_strides_[perm];
                ret.packed_shape_and_strides_[i + size] = packed_shape_and_strides_[perm + size];
            }
            return ret;
        }
        template<typename Integer>
        Self permuteWithWrapping(ArrayRef<Integer> perms) const {
            int size = ndim();
            Self ret;
            ret.packed_shape_and_strides_.resize(packed_shape_and_strides_.size());
            for (int i = 0; i < perms.size(); ++i) {
                int perm = perms[i];
                perm = perm >= 0 ? perm : perm + size;
                ret.packed_shape_and_strides_[i] = packed_shape_and_strides_[perm];
                ret.packed_shape_and_strides_[i + size] = packed_shape_and_strides_[perm + size];
            }
            return ret;
        }

         bool hasZeroStride() const {
            int size = ndim();
            const int64_t* strides = stride_data();
            for (size_t i = 0; i < size; ++i)
                if (strides[i] == 0)
                    return true;
            return false;
        }

         size_t tensorSize() const {
            int size = ndim();
            size_t ret = 1;
            for (int i = 0; i < size; ++i)
                ret *= packed_shape_and_strides_[i];
            return ret;
        }

         int ndim() const {
            return (packed_shape_and_strides_.size() >> 1);
        }

         const int64_t* shape_data() const {
            return packed_shape_and_strides_.data();
        }
         const int64_t* shape_begin() const {
            return shape_data();
        }
         const int64_t* shape_end() const {
            return packed_shape_and_strides_.data() + (packed_shape_and_strides_.size() >> 1);
        }
         const int64_t* stride_data() const {
            return packed_shape_and_strides_.data() + (packed_shape_and_strides_.size() >> 1);
        }
         const int64_t* stride_begin() const {
            return stride_data();
        }
         const int64_t* stride_end() const {
            return packed_shape_and_strides_.end();
        }

         int64_t* shape_data() {
            return packed_shape_and_strides_.data();
        }
         int64_t* shape_begin() {
            return shape_data();
        }
         int64_t* shape_end() {
            return packed_shape_and_strides_.data() + (packed_shape_and_strides_.size() >> 1);
        }
         int64_t* stride_data() {
            return packed_shape_and_strides_.data() + (packed_shape_and_strides_.size() >> 1);
        }
         int64_t* stride_begin() {
            return stride_data();
        }
         int64_t* stride_end() {
            return packed_shape_and_strides_.end();
        }

         void set_stride(LongArrayView stride) {
            assert(stride.size() == ndim());
            int dim = ndim();
            std::memcpy(shape_begin() + dim, stride.data(), dim * sizeof(int64_t));
        }
         void set_shape(LongArrayView shape) {
            assert(shape.size() == ndim());
            int dim = ndim();
            std::memcpy(shape_begin() , shape.data(), dim * sizeof(int64_t));
        }

         bool operator ==(const Self& y)const {
            return packed_shape_and_strides_ == y.packed_shape_and_strides_;
        }
         bool operator != (const Self& y)const {
            return !(*this == y);
        }

         void resize(size_t size) {
            packed_shape_and_strides_.resize(size);  
        }

    private:
        TensorShapeAndStride_t packed_shape_and_strides_;
    };



}//end megu