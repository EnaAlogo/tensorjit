#pragma warning( disable : 4244 ) 
#pragma once
#include <iostream>
#include "types.h"
#include "../common/export.h"

namespace megu {
     

#define ALIGNMENT(x) alignas(alignof(x))
#define ALIGNAS_SCALARTYPES ALIGNMENT(int8_t) ALIGNMENT(int32_t) ALIGNMENT(int64_t) ALIGNMENT(megu::half)\
    ALIGNMENT(megu::bfloat16) ALIGNMENT(float) ALIGNMENT(double) ALIGNMENT(std::complex<float>)\
    ALIGNMENT(std::complex<double>) 

    struct MEGU_API Scalar {  
        Scalar(float x) : type(dtype_t::Float) { new(value) float(x); }; 
        Scalar(int32_t x) : type(dtype_t::Int32) { new(value) int32_t(x); }; 
        Scalar(int64_t x) : type(dtype_t::Int64) { new(value) int64_t(x); };
        Scalar(int8_t x) : type(dtype_t::Int8) { new(value) int8_t(x); };
        Scalar(bool x) : type(dtype_t::Bool) { new(value) bool(x); };
        Scalar(double x) : type(dtype_t::Double) { new(value) double(x); };
        Scalar(megu::half x) : type(dtype_t::Half) { new(value) megu::half(x); };
        Scalar(bfloat16 x) : type(dtype_t::BFloat16) { new(value) bfloat16(x); };
        Scalar(std::complex<float> x) : type(dtype_t::Complex64) { new(value) std::complex<float>(x); };
        Scalar(const std::complex<double>& x) : type(dtype_t::Complex128) { new(value) std::complex<double>(x); };

        constexpr operator std::complex<float>() const { return SafeCast<std::complex<float>>(); }
        constexpr operator std::complex<double>() const { return SafeCast<std::complex<double>>(); }
        operator megu::half() const { return SafeCast<megu::half>(); }
        operator double() const { return SafeCast<double>(); }
        constexpr operator bool() const { return SafeCast<bool>(); }
        operator bfloat16() const { return SafeCast<bfloat16>(); }
        constexpr operator int8_t() const { return SafeCast<int8_t>(); }
        constexpr operator int64_t() const { return SafeCast<int64_t>(); }
        constexpr operator int32_t() const { return SafeCast<int32_t>(); }
        constexpr operator float() const { return SafeCast<float>(); }


         constexpr megu::dtype_t dtype() const { return type; };

         bool is_zero() const;

         Scalar neg() const;

         Scalar reciprocal() const;

         Scalar astype(dtype_t t)const;

         Scalar conj()const;

         void* mutable_raw_ptr() { return &value[0]; }

         const void* data() const { return &value[0]; }

    private:
        
        template<typename T>
        constexpr T SafeCast() const {
            switch (type) {
            case Int32:
                return T(*reinterpret_cast<const int32_t*>(value));
            case Int64:
                return T(*reinterpret_cast<const int64_t*>(value));
            case Int8:
                return T(*reinterpret_cast<const int8_t*>(value));
            case Bool:
                return T(*reinterpret_cast<const bool*>(value));
            case Float:
                return T(*reinterpret_cast<const float*>(value));
            case Double:
                return T(*reinterpret_cast<const double*>(value));
            case Half:
                return T(*reinterpret_cast<const megu::half*>(value));
            case BFloat16:
                return T(*reinterpret_cast<const megu::bfloat16*>(value));
            case Complex64:
                return T(reinterpret_cast<const std::complex<float>*>(value)->real());
            case Complex128:   // going from complex to real is .real discarding the imaginary(?)
                return T(reinterpret_cast<const std::complex<double>*>(value)->real()); 
            default:
                return T(0);
            }
        };

        template<>
        constexpr std::complex<float> SafeCast() const {
            
            switch (type) {
            case Int32:
                return  std::complex<float>(*reinterpret_cast<const int32_t*>(value));
            case Int64:
                return  std::complex<float>(*reinterpret_cast<const int64_t*>(value));
            case Int8:
                return  std::complex<float>(*reinterpret_cast<const int8_t*>(value));
            case Bool:
                return  std::complex<float>(*reinterpret_cast<const bool*>(value));
            case Float:
                return  std::complex<float>(*reinterpret_cast<const float*>(value));
            case Double:
                return  std::complex<float>(*reinterpret_cast<const double*>(value)); 
            case Half:
                return std::complex<float>(*reinterpret_cast<const megu::half*>(value));
            case BFloat16:
                return std::complex<float>(*reinterpret_cast<const megu::bfloat16*>(value));
            case Complex64:
                return *reinterpret_cast<const std::complex<float>*>(value);
            case Complex128: {
                const std::complex<double>*
                    _t = reinterpret_cast<const std::complex<double>*>(value);
                return std::complex<float>(_t->real(), _t->imag());
            }
            default:
                return std::complex<float>(0);
            }
        };

        template<>
        constexpr std::complex<double> SafeCast() const {
            switch (type) {
            case Int32:
                return  std::complex<double>(*reinterpret_cast<const int32_t*>(value));
            case Int64:
                return  std::complex<double>(*reinterpret_cast<const int64_t*>(value));
            case Int8:
                return  std::complex<double>(*reinterpret_cast<const int8_t*>(value));
            case Bool:
                return  std::complex<double>(*reinterpret_cast<const bool*>(value));
            case Float:
                return  std::complex<double>(*reinterpret_cast<const float*>(value));
            case Double:
                return  std::complex<double>(*reinterpret_cast<const double*>(value));
            case Half:
                return  std::complex<double>(*reinterpret_cast<const megu::half*>(value));
            case BFloat16:
                return  std::complex<double>(*reinterpret_cast<const megu::bfloat16*>(value));
            case Complex64: {
                const std::complex<float>*  _t = reinterpret_cast<const std::complex<float> *>(value);
                return  std::complex<double>(_t->real(), _t->imag());
            }
            case Complex128:
                return *reinterpret_cast<const  std::complex<double> *>(value);
            default:
                return  std::complex<double>(0);
            }
        };

        template<>
         megu::half SafeCast() const {
            switch (type) {
            case Int32:
                return megu::half(float(*reinterpret_cast<const int32_t*>(value)));
            case Int64:
                return megu::half(float(*reinterpret_cast<const int64_t*>(value)));
            case Int8:
                return megu::half(float(*reinterpret_cast<const int8_t*>(value)));
            case Bool:
                return megu::half(float(*reinterpret_cast<const bool*>(value)));
            case Float:
                return megu::half(*reinterpret_cast<const float*>(value));
            case Double:
                return megu::half(*reinterpret_cast<const double*>(value));
            case Half:
                return megu::half(*reinterpret_cast<const megu::half*>(value));
            case BFloat16:
                return megu::half(*reinterpret_cast<const megu::bfloat16*>(value));
            case Complex64:
                return megu::half(reinterpret_cast<const  std::complex<float> *>(value)->real());
            case Complex128:
                return megu::half(reinterpret_cast<const  std::complex<double> *>(value)->real());
            default:
                return megu::half(0.f);
            }
        };

        template<>
         bfloat16 SafeCast() const {
            switch (type) {
            case Int32:
                return bfloat16(float(*reinterpret_cast<const int32_t*>(value)));
            case Int64:
                return bfloat16(float(*reinterpret_cast<const int64_t*>(value)));
            case Int8:
                return bfloat16(float(*reinterpret_cast<const int8_t*>(value)));
            case Bool:
                return bfloat16(float(*reinterpret_cast<const bool*>(value)));
            case Float:
                return bfloat16(*reinterpret_cast<const float*>(value));
            case Double:
                return bfloat16(*reinterpret_cast<const double*>(value));
            case Half:
                return bfloat16(*reinterpret_cast<const megu::half*>(value));
            case BFloat16:
                return bfloat16(*reinterpret_cast<const megu::bfloat16*>(value));
            case Complex64:
                return bfloat16(reinterpret_cast<const  std::complex<float> *>(value)->real());
            case Complex128:
                return bfloat16(reinterpret_cast<const  std::complex<double> *>(value)->real());
            default:
                return bfloat16(0.f);
            }
        };


   
        ALIGNAS_SCALARTYPES unsigned char value[sizeof(std::complex<double>)] = {  };
        dtype_t type;
    };

#undef ALIGNAS_SCALARTYPES
#undef ALIGNMENT

};
