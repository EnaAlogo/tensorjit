#pragma once
#include <complex>
#include "../scalar_cvt.hpp"

namespace megu {

    using Device = int;

    enum dtype_t : int8_t 
    {
        Bool,
        Uint8,
        Int8,
        Uint16,
        Int16,
        Uint32,
        Int32,
        Uint64,
        Int64,
        //add float8
        Half,
        BFloat16,
        Float,
        Double,
        //add half/quarter precision complex
        Complex64,
        Complex128,
    };

    namespace detail {
        template<typename T>
        struct primitive_to_dtype;

        template<>
        struct primitive_to_dtype<float> {
            constexpr static auto value = Float;
        };
        template<>
        struct primitive_to_dtype<double> {
            constexpr static auto value = Double;
        };
        template<>
        struct primitive_to_dtype<int32_t> {
            constexpr static auto value = Int32;
        };
        template<>
        struct primitive_to_dtype<int64_t> {
            constexpr static auto value = Int64;
        };
        template<>
        struct primitive_to_dtype<bool> {
            constexpr static auto value = Bool;
        };
        template<>
        struct primitive_to_dtype<int8_t> {
            constexpr static auto value = Int8;
        };

        template<>
        struct primitive_to_dtype<uint8_t> {
            constexpr static auto value = Uint8;
        };

        template<>
        struct primitive_to_dtype<int16_t> {
            constexpr static auto value = Int16;
        };

        template<>
        struct primitive_to_dtype<uint16_t> {
            constexpr static auto value = Uint16;
        };
        template<>
        struct primitive_to_dtype<uint32_t> {
            constexpr static auto value = Uint32;
        };
        template<>
        struct primitive_to_dtype<uint64_t> {
            constexpr static auto value = Uint64;
        };

        template<>
        struct primitive_to_dtype<std::complex<float>> {
            constexpr static auto value = Complex64;
        };
        template<>
        struct primitive_to_dtype<std::complex<double>> {
            constexpr static auto value = Complex128;
        };

        template<>
        struct primitive_to_dtype<megu::half> {
            constexpr static auto value = Half;
        };
        template<>
        struct primitive_to_dtype<megu::bfloat16> {
            constexpr static auto value = BFloat16;
        };


        template<dtype_t type>
        struct underlying_type;

        template<>
        struct underlying_type<dtype_t::Float> {
            using Type = float;
        };

        template<>
        struct underlying_type<dtype_t::Double> {
            using Type = double;
        };
        template<>
        struct underlying_type<dtype_t::Int32> {
            using Type = int32_t;
        };
        template<>
        struct underlying_type<dtype_t::Int64> {
            using Type = int64_t;
        };
        template<>
        struct underlying_type<dtype_t::Bool> {
            using Type = bool;
        };
        template<>
        struct underlying_type<dtype_t::Int8> {
            using Type = int8_t;
        };
        template<>
        struct underlying_type<dtype_t::Complex64> {
            using Type = std::complex<float>;
        };
        template<>
        struct underlying_type<dtype_t::Complex128> {
            using Type = std::complex<double>;
        };
        template<>
        struct underlying_type<dtype_t::Half> {
            using Type = megu::half;
        };
        template<>
        struct underlying_type<dtype_t::BFloat16> {
            using Type = megu::bfloat16;
        };
        template<>
        struct underlying_type<dtype_t::Uint16> {
            using Type = uint16_t;
        };
        template<>
        struct underlying_type<dtype_t::Uint32> {
            using Type = uint32_t;
        };
        template<>
        struct underlying_type<dtype_t::Uint64> {
            using Type = uint64_t;
        };
        template<>
        struct underlying_type<dtype_t::Uint8> {
            using Type = int8_t;
        };

        template<>
        struct underlying_type<dtype_t::Int16> {
            using Type = int16_t;
        };


        template<dtype_t type>
        using underlying_type_t = underlying_type<type>::Type;

        template<typename T>
        struct acc_type { using type = T; };
        template<>
        struct acc_type<megu::half> { using type = float; };
        template<>
        struct acc_type<megu::bfloat16> { using type = float; };
        template<>
        struct acc_type<int32_t> { using type = int64_t; };
        template<>
        struct acc_type<int8_t> { using type = int64_t; };

        template<typename T>
        using acc_type_t = acc_type<T>::type;

        template<typename T>
        struct real_type { using type = T; };
        
        template<typename T>
        struct real_type<std::complex<T>> { using type = T; }; 

        template<typename T>
        using real_type_t = real_type<T>::type;


        constexpr inline size_t type_size(dtype_t type)
        {
            switch (type)
            {
            case Float:
            case Int32:
            case Uint32:
                return sizeof(float);
            case Double:
            case Complex64:
            case Int64:
            case Uint64:
                return sizeof(std::complex<float>);
            case Int16:
            case Uint16:
            case BFloat16:
            case Half:
                return sizeof(uint16_t);
            case Bool:
            case Int8:
            case Uint8:
                return sizeof(char);
            case Complex128:
                return sizeof(std::complex<double>);
            default:
                throw std::runtime_error("Invalid type");
            }
        }


        constexpr inline bool is_complex(const dtype_t t) { return t >= Complex64; };
        constexpr inline bool is_floating_point(const dtype_t t) { return t < Complex64 && t >= Half; };
        constexpr inline bool is_integral(const dtype_t t, bool include_bool) {
            bool is_intg = t <= Int64;
            return  is_intg || (include_bool && t == dtype_t::Bool);
        };

        constexpr inline bool is_reduced_fp(const dtype_t t) { return t == Half || t == BFloat16; }

        
        constexpr inline dtype_t real_type(const dtype_t dtype) {
            if (!is_complex(dtype)) {
                return dtype;
            }
            return dtype == Complex64 ? Float : Double;
        }

        constexpr inline dtype_t to_signed(const dtype_t dtype) {
            switch (dtype)
            {
            case Uint8:
                return Int8;
            case Uint16:
                return Int16;
            case Uint32:
                return Int32;
            case Uint64:
                return Int64;
            default:
                return dtype;
                break;
            }
        }

        constexpr inline bool can_cast(const dtype_t from, const dtype_t to) {
            if (is_complex(from) && !is_complex(to)) {
                return false;
            }

            if (is_floating_point(from) && is_integral(to, false)) {
                return false;
            }
            if (from != dtype_t::Bool && to == dtype_t::Bool) {
                return false;
            }
            return true;
        }

        constexpr inline dtype_t math_type(const dtype_t dtype) {
            if (is_integral(dtype, false)) {
                return Int64;
            }
            if (is_reduced_fp(dtype)) {
                return Float;
            }
            return dtype;
        }

        inline dtype_t promote_type(dtype_t r, dtype_t l) {
            constexpr dtype_t i32 = dtype_t::Int32;
            constexpr dtype_t i8 = dtype_t::Int8;
            constexpr dtype_t b = dtype_t::Bool;
            constexpr dtype_t i64 = dtype_t::Int64;
            constexpr dtype_t f32 = dtype_t::Float;
            constexpr dtype_t f64 = dtype_t::Double;
            constexpr dtype_t c = dtype_t::Complex64;
            constexpr dtype_t z = dtype_t::Complex128;
            constexpr dtype_t f16 = dtype_t::Half;
            constexpr dtype_t b16 = dtype_t::BFloat16;

            static constexpr dtype_t rules[10][10] = {
                {i8 , i8 , i32 , i64 , f16 , b16 , f32 , f64 , c , z},
                {i8 , i8 , i32 , i64 , f16 , b16 , f32 , f64 , c , z},
                {i32 , i32 , i32 , i64 , f16 , b16 , f32 , f64 , c , z},
                {i64 , i64 , i64 , i64 , f16 , b16 , f32 , f64 , c , z},
                {f16 , f16 , f16 , f16 , f16 , b16 , f32 , f64 , c , z},
                {b16 , b16 , b16 , b16 , b16 , b16 , f32 , f64 , c , z},
                {f32 , f32 , f32 , f32 , f32 , f32 , f32 , f64 , c , z},
                {f64 , f64 , f64 , f64 , f64 , f64 , f64 , f64 , c , z},
                {c , c , c , c , c , c , c , c , c , z},
                {z , z , z , z , z , z , z , z ,  z , z}
            };

            return rules[static_cast<int>(r)][static_cast<int>(l)];
        }


    }//end detail
}//end megu