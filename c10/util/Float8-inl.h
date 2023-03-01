#pragma once

#include <c10/macros/Macros.h>
#include <cstring>
#include <limits>

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

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE Float8::Float8(float value)
    :
#if defined(__CUDA_ARCH__)// || defined(__HIP_DEVICE_COMPILE__)
      x(reinterpret_cast<unsigned char>(__nv_cvt_float_to_fp8 (value,__NV_E5M2,__NV_SATFINITE)))
// #elif defined(__SYCL_DEVICE_ONLY__)
//       x(sycl::bit_cast<uint16_t>(sycl::half(value)))
#else
      x(detail::fp8_ieee_from_fp32_value(value))
#endif
{
}

/// Implicit conversions

inline C10_HOST_DEVICE Float8::operator float() const {
#if defined(__CUDA_ARCH__)// || defined(__HIP_DEVICE_COMPILE__)
  return float(*reinterpret_cast<const __nv_fp8_e5m2*>(&x));
// #elif defined(__SYCL_DEVICE_ONLY__)
//   return float(sycl::bit_cast<sycl::half>(x));
#else
  return detail::fp8_ieee_to_fp32_value(x);
#endif
}

#if defined(__CUDACC__)// || defined(__HIPCC__)
inline C10_HOST_DEVICE Float8::Float8(const __nv_fp8_e5m2& value) {
  x = *reinterpret_cast<const unsigned char*>(&value);
}
inline C10_HOST_DEVICE Float8::operator __nv_fp8_e5m2() const {
  return *reinterpret_cast<const __nv_fp8_e5m2*>(&x);
}
#endif

// #ifdef SYCL_LANGUAGE_VERSION
// inline C10_HOST_DEVICE Half::Half(const sycl::half& value) {
//   x = *reinterpret_cast<const unsigned short*>(&value);
// }
// inline C10_HOST_DEVICE Half::operator sycl::half() const {
//   return *reinterpret_cast<const sycl::half*>(&x);
// }
// #endif

// CUDA intrinsics

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || \
    (defined(__clang__) && defined(__CUDA__))
inline __device__ Float8 __ldg(const Float8* ptr) {
  return __ldg(reinterpret_cast<const __nv_fp8_e5m2*>(ptr));
}
#endif

/// Arithmetic

inline C10_HOST_DEVICE Float8 operator+(const Float8& a, const Float8& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8 operator-(const Float8& a, const Float8& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8 operator*(const Float8& a, const Float8& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8 operator/(const Float8& a, const Float8& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8 operator-(const Float8& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
  return __hneg(a);
// #elif defined(__SYCL_DEVICE_ONLY__)
//   return -sycl::bit_cast<sycl::half>(a);
#else
  return -static_cast<float>(a);
#endif
}

inline C10_HOST_DEVICE Float8& operator+=(Float8& a, const Float8& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Float8& operator-=(Float8& a, const Float8& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Float8& operator*=(Float8& a, const Float8& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Float8& operator/=(Float8& a, const Float8& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Float8 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Float8 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Float8 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Float8 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Float8 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Float8 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Float8 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Float8 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Float8 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Float8 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, Float8 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, Float8 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, Float8 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, Float8 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8 operator+(Float8 a, int b) {
  return a + static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator-(Float8 a, int b) {
  return a - static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator*(Float8 a, int b) {
  return a * static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator/(Float8 a, int b) {
  return a / static_cast<Float8>(b);
}

inline C10_HOST_DEVICE Float8 operator+(int a, Float8 b) {
  return static_cast<Float8>(a) + b;
}
inline C10_HOST_DEVICE Float8 operator-(int a, Float8 b) {
  return static_cast<Float8>(a) - b;
}
inline C10_HOST_DEVICE Float8 operator*(int a, Float8 b) {
  return static_cast<Float8>(a) * b;
}
inline C10_HOST_DEVICE Float8 operator/(int a, Float8 b) {
  return static_cast<Float8>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8 operator+(Float8 a, int64_t b) {
  return a + static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator-(Float8 a, int64_t b) {
  return a - static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator*(Float8 a, int64_t b) {
  return a * static_cast<Float8>(b);
}
inline C10_HOST_DEVICE Float8 operator/(Float8 a, int64_t b) {
  return a / static_cast<Float8>(b);
}

inline C10_HOST_DEVICE Float8 operator+(int64_t a, Float8 b) {
  return static_cast<Float8>(a) + b;
}
inline C10_HOST_DEVICE Float8 operator-(int64_t a, Float8 b) {
  return static_cast<Float8>(a) - b;
}
inline C10_HOST_DEVICE Float8 operator*(int64_t a, Float8 b) {
  return static_cast<Float8>(a) * b;
}
inline C10_HOST_DEVICE Float8 operator/(int64_t a, Float8 b) {
  return static_cast<Float8>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Half to float.

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;//3
  static constexpr int digits10 = 1;//1
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr c10::Float8 min() {
    return c10::Float8(0x04, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 lowest() {
    return c10::Float8(0xFB, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 max() {
    return c10::Float8(0x7B, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 epsilon() {
    return c10::Float8(0x14, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 round_error() {
    return c10::Float8(0x38, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 infinity() {
    return c10::Float8(0x7C, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 quiet_NaN() {
    return c10::Float8(0x7E, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 signaling_NaN() {
    return c10::Float8(0x7D, c10::Float8::from_bits());
  }
  static constexpr c10::Float8 denorm_min() {
    return c10::Float8(0x01, c10::Float8::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
