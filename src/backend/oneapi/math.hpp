/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <common/half.hpp>
#include <af/defines.h>

#include <backend.hpp>
#include <types.hpp>

#include <algorithm>
#include <complex>
#include <limits>

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#else
/* Other */
#endif

namespace arrayfire {
namespace oneapi {

template<typename T>
static inline T abs(T val) {
    return std::abs(val);
}
template<typename T>
static inline T min(T lhs, T rhs) {
    return std::min(lhs, rhs);
}
template<typename T>
static inline T max(T lhs, T rhs) {
    return std::max(lhs, rhs);
}

template<typename T>
static inline T division(T lhs, double rhs) {
    return lhs / rhs;
}
cfloat division(cfloat lhs, double rhs);
cdouble division(cdouble lhs, double rhs);

template<>
inline cfloat max<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
inline cdouble max<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template<>
inline cfloat min<cfloat>(cfloat lhs, cfloat rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<>
inline cdouble min<cdouble>(cdouble lhs, cdouble rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template<typename T>
static T scalar(double val) {
    return (T)(val);
}

template<>
inline cfloat scalar<cfloat>(double val) {
    cfloat cval(static_cast<float>(val));
    // cval.real() = (float)val;
    // cval.imag() = 0;
    return cval;
}

template<>
inline cdouble scalar<cdouble>(double val) {
    cdouble cval(val);
    return cval;
}

template<typename To, typename Ti>
static To scalar(Ti real, Ti imag) {
    To cval(real, imag);
    return cval;
}

template<typename T>
inline T maxval() {
    return std::numeric_limits<T>::max();
}
template<typename T>
inline T minval() {
    return std::numeric_limits<T>::min();
}
template<>
inline float maxval() {
    return std::numeric_limits<float>::infinity();
}
template<>
inline double maxval() {
    return std::numeric_limits<double>::infinity();
}

template<>
inline arrayfire::common::half maxval() {
    return std::numeric_limits<arrayfire::common::half>::infinity();
}

template<>
inline float minval() {
    return -std::numeric_limits<float>::infinity();
}

template<>
inline double minval() {
    return -std::numeric_limits<double>::infinity();
}
template<>
inline arrayfire::common::half minval() {
    return -std::numeric_limits<arrayfire::common::half>::infinity();
}

template<typename T>
static inline T real(T in) {
    return std::real(in);
}

template<typename T>
static inline T imag(T in) {
    return std::imag(in);
}

inline arrayfire::common::half operator+(arrayfire::common::half lhs,
                                         arrayfire::common::half rhs) noexcept {
    return arrayfire::common::half(static_cast<float>(lhs) +
                                   static_cast<float>(rhs));
}
}  // namespace oneapi
}  // namespace arrayfire

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic pop
#else
/* Other */
#endif
