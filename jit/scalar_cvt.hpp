//type conversions from c10 

/*
From PyTorch :

Copyright(c) 2016 - Facebook, Inc(Adam Paszke)
Copyright(c) 2014 - Facebook, Inc(Soumith Chintala)
Copyright(c) 2011 - 2014 Idiap Research Institute(Ronan Collobert)
Copyright(c) 2012 - 2014 Deepmind Technologies(Koray Kavukcuoglu)
Copyright(c) 2011 - 2012 NEC Laboratories America(Koray Kavukcuoglu)
Copyright(c) 2011 - 2013 NYU(Clement Farabet)
Copyright(c) 2006 - 2010 NEC Laboratories America(Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright(c) 2006      Idiap Research Institute(Samy Bengio)
Copyright(c) 2001 - 2004 Idiap Research Institute(Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2 :

Copyright(c) 2016 - present, Facebook Inc.All rights reserved.

All contributions by Facebook :
Copyright(c) 2016 Facebook Inc.

All contributions by Google :
Copyright(c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia :
Copyright(c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain :
Copyright 2019 - 2020 Kakao Brain

All contributions by Cruise LLC :
Copyright(c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe :
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions :
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe : each contributor holds
copyright over their contributions to Caffe2.The project versioning records
all such contribution and copyright details.If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met :

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and /or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
and IDIAP Research Institute nor the names of its contributors may be
used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>


namespace megu {


    namespace detail {

        //referece: https://github.com/pytorch/pytorch/blob/main/c10/util/Half.h 
        //and https://github.com/pytorch/pytorch/blob/main/c10/util/BFloat16.h

        inline float fp32_from_bits(uint32_t w) {
            union {
                uint32_t as_bits;
                float as_value;
            } fp32 = { w };
            return fp32.as_value;
        }

        inline uint32_t fp32_to_bits(float f) {
            union {
                float as_value;
                uint32_t as_bits;
            } fp32 = { f };
            return fp32.as_bits;
        }


        inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
            /*
             * Extend the half-precision floating-point number to 32 bits and shift to the
             * upper part of the 32-bit word:
             *      +---+-----+------------+-------------------+
             *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
             *      +---+-----+------------+-------------------+
             * Bits  31  26-30    16-25            0-15
             *
             * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
             * - zero bits.
             */
            const uint32_t w = (uint32_t)h << 16;
            /*
             * Extract the sign of the input number into the high bit of the 32-bit word:
             *
             *      +---+----------------------------------+
             *      | S |0000000 00000000 00000000 00000000|
             *      +---+----------------------------------+
             * Bits  31                 0-31
             */
            const uint32_t sign = w & UINT32_C(0x80000000);
            /*
             * Extract mantissa and biased exponent of the input number into the bits 0-30
             * of the 32-bit word:
             *
             *      +---+-----+------------+-------------------+
             *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
             *      +---+-----+------------+-------------------+
             * Bits  30  27-31     17-26            0-16
             */
            const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
            /*
             * Renorm shift is the number of bits to shift mantissa left to make the
             * half-precision number normalized. If the initial number is normalized, some
             * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
             * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
             * that if we shift denormalized nonsign by renorm_shift, the unit bit of
             * mantissa will shift into exponent, turning the biased exponent into 1, and
             * making mantissa normalized (i.e. without leading 1).
             */
#ifdef _MSC_VER
            unsigned long nonsign_bsr;
            _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
            uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
            uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
            renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
            /*
             * Iff half-precision number has exponent of 15, the addition overflows
             * it into bit 31, and the subsequent shift turns the high 9 bits
             * into 1. Thus inf_nan_mask == 0x7F800000 if the half-precision number
             * had exponent of 15 (i.e. was NaN or infinity) 0x00000000 otherwise
             */
            const int32_t inf_nan_mask =
                ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
            /*
             * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
             * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
             * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
             * 0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
             * 0x00000000 otherwise
             */
            const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
            /*
             * 1. Shift nonsign left by renorm_shift to normalize it (if the input
             * was denormal)
             * 2. Shift nonsign right by 3 so the exponent (5 bits originally)
             * becomes an 8-bit field and 10-bit mantissa shifts into the 10 high
             * bits of the 23-bit mantissa of IEEE single-precision number.
             * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
             * different in exponent bias (0x7F for single-precision number less 0xF
             * for half-precision number).
             * 4. Subtract renorm_shift from the exponent (starting at bit 23) to
             * account for renormalization. As renorm_shift is less than 0x70, this
             * can be combined with step 3.
             * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the
             * input was NaN or infinity.
             * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent
             * into zero if the input was zero.
             * 7. Combine with the sign of the input number.
             */
            return sign |
                ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
                    inf_nan_mask) &
                    ~zero_mask);
        }


        inline float fp16_ieee_to_fp32_value(uint16_t h) {
            /*
             * Extend the half-precision floating-point number to 32 bits and shift to the
             * upper part of the 32-bit word:
             *      +---+-----+------------+-------------------+
             *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
             *      +---+-----+------------+-------------------+
             * Bits  31  26-30    16-25            0-15
             *
             * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
             * - zero bits.
             */
            const uint32_t w = (uint32_t)h << 16;
            /*
             * Extract the sign of the input number into the high bit of the 32-bit word:
             *
             *      +---+----------------------------------+
             *      | S |0000000 00000000 00000000 00000000|
             *      +---+----------------------------------+
             * Bits  31                 0-31
             */
            const uint32_t sign = w & UINT32_C(0x80000000);
            /*
             * Extract mantissa and biased exponent of the input number into the high bits
             * of the 32-bit word:
             *
             *      +-----+------------+---------------------+
             *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
             *      +-----+------------+---------------------+
             * Bits  27-31    17-26            0-16
             */
            const uint32_t two_w = w + w;

            /*
             * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
             * mantissa and exponent of a single-precision floating-point number:
             *
             *       S|Exponent |          Mantissa
             *      +-+---+-----+------------+----------------+
             *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
             *      +-+---+-----+------------+----------------+
             * Bits   | 23-31   |           0-22
             *
             * Next, there are some adjustments to the exponent:
             * - The exponent needs to be corrected by the difference in exponent bias
             * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
             * - Inf and NaN values in the inputs should become Inf and NaN values after
             * conversion to the single-precision number. Therefore, if the biased
             * exponent of the half-precision input was 0x1F (max possible value), the
             * biased exponent of the single-precision output must be 0xFF (max possible
             * value). We do this correction in two steps:
             *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
             * below) rather than by 0x70 suggested by the difference in the exponent bias
             * (see above).
             *   - Then we multiply the single-precision result of exponent adjustment by
             * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
             * necessary exponent adjustment by 0x70 due to difference in exponent bias.
             *     The floating-point multiplication hardware would ensure than Inf and
             * NaN would retain their value on at least partially IEEE754-compliant
             * implementations.
             *
             * Note that the above operations do not handle denormal inputs (where biased
             * exponent == 0). However, they also do not operate on denormal inputs, and
             * do not produce denormal results.
             */
            constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
            // const float exp_scale = 0x1.0p-112f;
            constexpr uint32_t scale_bits = (uint32_t)15 << 23;

            const float exp_scale = std::bit_cast<float>(scale_bits);
            const float normalized_value =
                fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

            /*
             * Convert denormalized half-precision inputs into single-precision results
             * (always normalized). Zero inputs are also handled here.
             *
             * In a denormalized number the biased exponent is zero, and mantissa has
             * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
             *
             *                  zeros           |  mantissa
             *      +---------------------------+------------+
             *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
             *      +---------------------------+------------+
             * Bits             10-31                0-9
             *
             * Now, remember that denormalized half-precision numbers are represented as:
             *    FP16 = mantissa * 2**(-24).
             * The trick is to construct a normalized single-precision number with the
             * same mantissa and thehalf-precision input and with an exponent which would
             * scale the corresponding mantissa bits to 2**(-24). A normalized
             * single-precision floating-point number is represented as: FP32 = (1 +
             * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
             * exponent is 126, a unit change in the mantissa of the input denormalized
             * half-precision number causes a change of the constructed single-precision
             * number by 2**(-24), i.e. the same amount.
             *
             * The last step is to adjust the bias of the constructed single-precision
             * number. When the input half-precision number is zero, the constructed
             * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
             * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
             * single-precision number to get the numerical equivalent of the input
             * half-precision number.
             */
            constexpr uint32_t magic_mask = UINT32_C(126) << 23;
            constexpr float magic_bias = 0.5f;
            const float denormalized_value =
                fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

            /*
             * - Choose either results of conversion of input as a normalized number, or
             * as a denormalized number, depending on the input exponent. The variable
             * two_w contains input exponent in bits 27-31, therefore if its smaller than
             * 2**27, the input is either a denormal number, or zero.
             * - Combine the result of conversion of exponent and mantissa with the sign
             * of the input number.
             */
            constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
            const uint32_t result = sign |
                (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                    : fp32_to_bits(normalized_value));
            return fp32_from_bits(result);
        }


        inline uint16_t fp16_ieee_from_fp32_value(float f) {
            // const float scale_to_inf = 0x1.0p+112f;
            // const float scale_to_zero = 0x1.0p-110f;
            constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
            constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;

            float const scale_to_inf = std::bit_cast<float>(scale_to_inf_bits);
            float const scale_to_zero = std::bit_cast<float>(scale_to_zero_bits);

#if defined(_MSC_VER) && _MSC_VER == 1916
            float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
#else
            float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
#endif

            const uint32_t w = fp32_to_bits(f);
            const uint32_t shl1_w = w + w;
            const uint32_t sign = w & UINT32_C(0x80000000);
            uint32_t bias = shl1_w & UINT32_C(0xFF000000);
            if (bias < UINT32_C(0x71000000)) {
                bias = UINT32_C(0x71000000);
            }

            base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
            const uint32_t bits = fp32_to_bits(base);
            const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
            const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
            const uint32_t nonsign = exp_bits + mantissa_bits;
            return static_cast<uint16_t>(
                (sign >> 16) |
                (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
        }

        constexpr inline float f32_from_bits(uint16_t src) {
            uint32_t tmp = src;
            tmp <<= 16;
            return std::bit_cast<float, uint32_t>(tmp);
        }

        constexpr inline uint16_t bits_from_f32(float src) {
            uint32_t res = std::bit_cast<float, uint32_t>(src);
            return res >> 16;
        }

        inline uint16_t round_to_nearest_even(float src) {
#if defined(_MSC_VER)
            if (isnan(src)) {
#else
            if (std::isnan(src)) {
#endif
                return UINT16_C(0x7FC0);
            }
            else {
                union {
                    uint32_t U32;
                    float F32;
                };

                F32 = src;
                uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
                return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
            }
            }

        }//end detail

    struct alignas(2) half {
        half(float x) { raw = detail::fp16_ieee_from_fp32_value(x); }
        operator float()const {
            return detail::fp16_ieee_to_fp32_value(raw);
        }

        uint16_t raw;
    };

    struct alignas(2) bfloat16 {
        bfloat16(float x) { raw = detail::round_to_nearest_even(x); }
        constexpr operator float()const {
            return detail::f32_from_bits(raw);
        }

        uint16_t raw;
    };

   
}//end megu