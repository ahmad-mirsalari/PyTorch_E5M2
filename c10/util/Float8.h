#pragma once

/// Defines the float8|e5m2 type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32, instead of using CUDA intrinsics.

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/complex.h>
#include <type_traits>

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cmath>
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <c10/util/Half.h>
#ifdef __CUDACC__
#include <cuda_fp8.h>
#endif

// #ifdef __HIPCC__
// #include <hip/hip_fp16.h>
// #endif

// #if defined(CL_SYCL_LANGUAGE_VERSION)
// #include <CL/sycl.hpp> // for SYCL 1.2.1
// #elif defined(SYCL_LANGUAGE_VERSION)
// #include <sycl/sycl.hpp> // for SYCL 2020
// #endif

// Standard check for compiling CUDA with clang
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define C10_DEVICE_HOST_FUNCTION __device__ __host__
#else
#define C10_DEVICE_HOST_FUNCTION
#endif

#include <typeinfo> // operator typeid

namespace c10 {

namespace detail {

//beginning of comment
// C10_DEVICE_HOST_FUNCTION inline float fp32_from_bits(uint32_t w) {
// #if defined(__OPENCL_VERSION__)
//   return as_float(w);
// #elif defined(__CUDA_ARCH__)
//   return __uint_as_float((unsigned int)w);
// #elif defined(__INTEL_COMPILER)
//   return _castu32_f32(w);
// #else
//   union {
//     uint32_t as_bits;
//     float as_value;
//   } fp32 = {w};
//   return fp32.as_value;
// #endif
// }

// C10_DEVICE_HOST_FUNCTION inline uint32_t fp32_to_bits(float f) {
// #if defined(__OPENCL_VERSION__)
//   return as_uint(f);
// #elif defined(__CUDA_ARCH__)
//   return (uint32_t)__float_as_uint(f);
// #elif defined(__INTEL_COMPILER)
//   return _castf32_u32(f);
// #else
//   union {
//     float as_value;
//     uint32_t as_bits;
//   } fp32 = {f};
//   return fp32.as_bits;
// #endif
// }
// ending of comment

/*
 * Convert a 8-bit floating-point number in E5M2 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
 
 
inline uint32_t fp8_ieee_to_fp32_bits(uint8_t h) {
  /*
   * Extend the 8-bit floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+------+--+------------------------------+
   *      | S |EEEE E|MM| 0000 0000 0000 0000 0000 0000|
   *      +---+------+--+------------------------------+
   * Bits  31  26-30    24-25            0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 24;
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
   *      +---+------+--+------------------------------+
   *      | 0 |EEEE E|MM| 0000 0000 0000 0000 0000 0000|
   *      +---+------+--+------------------------------+
   * Bits  31  26-30    24-25            0-23
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * FP8 number normalized. If the initial number is normalized, some
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
   * Iff FP8 number has exponent of 15, the addition overflows
   * it into bit 31, and the subsequent shift turns the high 9 bits
   * into 1. Thus inf_nan_mask == 0x7F800000 if the FP8 number
   * had exponent of 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
   * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
   * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
   * 0xFFFFFFFF if the FP8 number was zero (+0.0h or -0.0h)
   * 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input
   * was denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally)
   * becomes an 8-bit field and 2-bit mantissa shifts into the 2 high
   * bits of the 23-bit mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF
   * for FP8 number).
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

/*
 * Convert a 8-bit floating-point number in IEEE FP8 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline float fp8_ieee_to_fp32_value(uint8_t h) {
  /*
   * Extend the 8-bit floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+------+--+------------------------------+
   *      | S |EEEE E|MM| 0000 0000 0000 0000 0000 0000|
   *      +---+------+--+------------------------------+
   * Bits  31  26-30    24-25            0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 24;
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
   *      +---+------+--+------------------------------+
   *      | 0 |EEEE E|MM| 0000 0000 0000 0000 0000 0000|
   *      +---+------+--+------------------------------+
   * Bits  31  26-30    24-25            0-23
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+--+--------------------------+
   *      |0|000|EEEEE|MM|0000 0000|0 0000 0000 0000|
   *      +-+---+-----+--+--------------------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and FP8 formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the FP8 input was 0x1F (max possible value), the
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
  float exp_scale_val;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized FP8 inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * two_w representation:
   *      +------+--+-------------------------------+
   *      |EEEE E|MM|0 0000 0000 0000 0000 0000 0000|
   *      +------+--+-------------------------------+
   * Bits  31  27 25-26            0-24
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-1 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM 0000 0000|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized FP8 numbers are represented as:
   *    FP8 = mantissa * 2**(-16).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and the FP8 input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * FP8 number causes a change of the constructud single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input FP8 number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * FP8 number.
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

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
 inline int find_digit(float number) {
 
    int power_of_10 = 1;
    while (number * power_of_10 != (int)(number * power_of_10)) {
        power_of_10 *= 10;
    }
    int integer_number = (int)(number * power_of_10);
    
     int tens_digit = (integer_number / 10) % 10; // extract the digit before the last one
 
    return tens_digit;
}
 
inline uint8_t fp8_ieee_from_fp32_value(float f) {

 /* constexpr uint32_t scale_to_inf_bits = (uint32_t)127 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)15 << 23;
  float scale_to_inf_val, scale_to_zero_val;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

#if defined(_MSC_VER) && _MSC_VER == 1916
  float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
#else
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
#endif

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFE000000);
  if (bias < UINT32_C(0x5E000000)) {
    bias = UINT32_C(0x5E000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x38000000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 10) & UINT32_C(0x0000001F);
  const uint32_t mantissa_bits = bits & UINT32_C(0x000003FF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<unsigned char>(
      (sign >> 24) |
      (shl1_w > UINT32_C(0xFE000000) ?  UINT8_C(0x7F) : nonsign));*/
      
  // const float scale_to_inf = 0x1.0p+112f;
  // const float scale_to_zero = 0x1.0p-110f;
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val, scale_to_zero_val;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

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
  const uint32_t exp_bits = (bits >> 21) & UINT32_C(0x0000007C);
  
  /*
  Rounding 
  */
   uint32_t trunc =UINT32_C(0x00000000) ;
  if ((bits & UINT32_C(0x000000FF)) > UINT32_C(0x00000081)){ // mantisa is greater than the half
    trunc = UINT32_C(0x00000001);
    printf("I am here in the 1");
  }
  else if ((bits & UINT32_C(0x000000FF)) < UINT32_C(0x00000080)){ // mantisa is less than the half
    trunc = UINT32_C(0x00000000);
    printf("I am here in the 2");
  }
  else if ((bits & UINT32_C(0x000000FF)) == UINT32_C(0x00000080)){ // mantisa is equal to the half
    
   if (static_cast<int>(find_digit(f)) % 2 == 0) { //f (static_cast<int>(x) % 2 == 0) { /* x is "even" */ }
     trunc = UINT32_C(0x00000000);
     printf("I am here in the 3");
   }
   else {
     trunc = UINT32_C(0x00000001);
     printf("I am here in the 34");
   }
  }
  /*
  End of Rounding
  */
  const uint32_t mantissa_bits = ((bits >> 8) & UINT32_C(0x0000000F)) + trunc; //+ ((bits >> 7) & UINT32_C(0x00000001))   ; 
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint8_t>(
      (sign >> 24) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT8_C(0x7E) : nonsign));
}

} // namespace detail

struct alignas(1) Float8 {
  unsigned char x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // HIP wants __host__ __device__ tag, CUDA does not
#if defined(USE_ROCM)
  C10_HOST_DEVICE Float8() = default;
#else
  Float8() = default;
#endif

  constexpr C10_HOST_DEVICE Float8(unsigned char bits, from_bits_t) : x(bits){};
  inline C10_HOST_DEVICE Float8(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_HOST_DEVICE Float8(const __nv_fp8_e5m2 & value);
  inline C10_HOST_DEVICE operator __nv_fp8_e5m2 () const;
#endif
// #ifdef SYCL_LANGUAGE_VERSION
//   inline C10_HOST_DEVICE Float8(const sycl::half& value);
//   inline C10_HOST_DEVICE operator sycl::half() const;
// #endif
};

// TODO : move to complex.h
template <>
struct alignas(2) complex<Float8> {
  Float8 real_;
  Float8 imag_;

  // Constructors
  complex() = default;
  // Float8 constructor is not constexpr so the following constructor can't
  // be constexpr
  C10_HOST_DEVICE explicit inline complex(const Float8& real, const Float8& imag)
      : real_(real), imag_(imag) {}
  C10_HOST_DEVICE inline complex(const c10::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  // Conversion operator
  inline C10_HOST_DEVICE operator c10::complex<float>() const {
    return {real_, imag_};
  }

  constexpr C10_HOST_DEVICE Float8 real() const {
    return real_;
  }
  constexpr C10_HOST_DEVICE Float8 imag() const {
    return imag_;
  }

  C10_HOST_DEVICE complex<Float8>& operator+=(const complex<Float8>& other) {
    real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
    return *this;
  }

  C10_HOST_DEVICE complex<Float8>& operator-=(const complex<Float8>& other) {
    real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
    return *this;
  }

  C10_HOST_DEVICE complex<Float8>& operator*=(const complex<Float8>& other) {
    auto a = static_cast<float>(real_);
    auto b = static_cast<float>(imag_);
    auto c = static_cast<float>(other.real());
    auto d = static_cast<float>(other.imag());
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }
};

//beginning of comment

// // In some versions of MSVC, there will be a compiler error when building.
// // C4146: unary minus operator applied to unsigned type, result still unsigned
// // C4804: unsafe use of type 'bool' in operation
// // It can be addressed by disabling the following warning.
// #ifdef _MSC_VER
// #pragma warning(push)
// #pragma warning(disable : 4146)
// #pragma warning(disable : 4804)
// #pragma warning(disable : 4018)
// #endif

// // The overflow checks may involve float to int conversion which may
// // trigger precision loss warning. Re-enable the warning once the code
// // is fixed. See T58053069.
// #ifdef __clang__
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wunknown-warning-option"
// #pragma GCC diagnostic ignored "-Wimplicit-int-float-conversion"
// #endif

// // bool can be converted to any type.
// // Without specializing on bool, in pytorch_linux_trusty_py2_7_9_build:
// // `error: comparison of constant '255' with boolean expression is always false`
// // for `f > limit::max()` below
// template <typename To, typename From>
// typename std::enable_if<std::is_same<From, bool>::value, bool>::type overflows(
//     From /*f*/) {
//   return false;
// }

// // skip isnan and isinf check for integral types
// template <typename To, typename From>
// typename std::enable_if<
//     std::is_integral<From>::value && !std::is_same<From, bool>::value,
//     bool>::type
// overflows(From f) {
//   using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
//   if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
//     // allow for negative numbers to wrap using two's complement arithmetic.
//     // For example, with uint8, this allows for `a - b` to be treated as
//     // `a + 255 * b`.
//     return greater_than_max<To>(f) ||
//         (c10::is_negative(f) && -static_cast<uint64_t>(f) > limit::max());
//   } else {
//     return c10::less_than_lowest<To>(f) || greater_than_max<To>(f);
//   }
// }

// template <typename To, typename From>
// typename std::enable_if<std::is_floating_point<From>::value, bool>::type
// overflows(From f) {
//   using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
//   if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
//     return false;
//   }
//   if (!limit::has_quiet_NaN && (f != f)) {
//     return true;
//   }
//   return f < limit::lowest() || f > limit::max();
// }

// #ifdef __clang__
// #pragma GCC diagnostic pop
// #endif

// #ifdef _MSC_VER
// #pragma warning(pop)
// #endif

// template <typename To, typename From>
// typename std::enable_if<is_complex<From>::value, bool>::type overflows(From f) {
//   // casts from complex to real are considered to overflow if the
//   // imaginary component is non-zero
//   if (!is_complex<To>::value && f.imag() != 0) {
//     return true;
//   }
//   // Check for overflow componentwise
//   // (Technically, the imag overflow check is guaranteed to be false
//   // when !is_complex<To>, but any optimizer worth its salt will be
//   // able to figure it out.)
//   return overflows<
//              typename scalar_value_type<To>::type,
//              typename From::value_type>(f.real()) ||
//       overflows<
//              typename scalar_value_type<To>::type,
//              typename From::value_type>(f.imag());
// }


//ending of comment
C10_API std::ostream& operator<<(std::ostream& out, const Float8& value);

} // namespace c10

#include <c10/util/Float8-inl.h> // IWYU pragma: keep
