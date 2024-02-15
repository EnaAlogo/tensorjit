#pragma once
#include "scalar.hpp"
#include "types.h"


namespace megu::cuda::jit {

    struct JitScalarArg{
        JitScalarArg(std::string name, Scalar const& value)
            :name_(std::move(name)), value_(value) {}

        std::string_view name()const { return name_; }

        Scalar const& value()const { return value_; }

        dtype_t dtype()const { return value_.dtype(); }

        void const* ptr()const { return value_.data(); }

        size_t type_size()const { return detail::type_size(dtype()); }

        Scalar astype(dtype_t type)const {
            return value_.astype(type);
        }
        JitScalarArg& iastype(dtype_t type) {
            value_ = value_.astype(type);
            return *this;
        }
    private:
        std::string const name_;
        Scalar value_;
    };


}//end megu::cuda::jit