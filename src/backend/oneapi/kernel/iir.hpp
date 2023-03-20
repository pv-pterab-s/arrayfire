/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
using local_accessor = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                      sycl::access::target::local>;
template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;


template<typename T>
T __mul(T a, T b) {
    return a * b;
}

template<typename T>
T __div(T a, T b) {
    return a / b;
}

template <typename T>
std::complex<T> __cconjf(std::complex<T> in) {
    return {in.real(), -in.imag()};
}

  template <typename T>
  std::complex<T> __mul(std::complex<T> lhs, std::complex<T> rhs) {
      return {
        lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
        lhs.real() * rhs.imag() + lhs.imag() * rhs.real()
      };
  }

template<typename T>
std::complex<T> __div(std::complex<T> lhs, std::complex<T> rhs) {
    auto den = (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
    std::complex<T> num  = __mul(lhs, __cconjf(rhs));
    return { num.real() / den, num.imag() / den };
}

template<typename T>
class iirCreateKernel {
public:
 iirCreateKernel(write_accessor<T> yptr, const KParam yinfo,
                 read_accessor<T> cptr, const KParam cinfo,
                 read_accessor<T> aptr, const KParam ainfo,
                 const int groups_y, local_accessor<T> s_z, local_accessor<T> s_a, local_accessor<T> s_y, const size_t MAX_A_SIZE)
   : yptr_(yptr)
     , yinfo_(yinfo)
     , cptr_(cptr)
     , cinfo_(cinfo)
     , aptr_(aptr)
     , ainfo_(ainfo)
     , groups_y_(groups_y) , s_z_(s_z) , s_a_(s_a) , s_y_(s_y) , MAX_A_SIZE_(MAX_A_SIZE) {}
  void operator()(sycl::nd_item<2> it) const {
      sycl::group g = it.get_group();

      const int idz = g.get_group_id(0);
      const int idw = g.get_group_id(1) / groups_y_;
      const int idy = g.get_group_id(1) - idw * groups_y_;

      const int tx    = it.get_local_id(0);
      const int num_a = ainfo_.dims[0];

      int y_off = idw * yinfo_.strides[3] + idz * yinfo_.strides[2] +
        idy * yinfo_.strides[1];
      int c_off = idw * cinfo_.strides[3] + idz * cinfo_.strides[2] +
        idy * cinfo_.strides[1];

#if BATCH_A
      int a_off = idw * ainfo_.strides[3] + idz * ainfo_.strides[2] +
        idy * ainfo_.strides[1];
#else
      int a_off = 0;
#endif

      T *d_y       = yptr_.get_pointer() + y_off;
      const  T *d_c = cptr_.get_pointer() + c_off + cinfo_.offset;
      const  T *d_a = aptr_.get_pointer() + a_off + ainfo_.offset;
      const int repeat      = (num_a + g.get_local_range(0) - 1) /
        g.get_local_range(0);

      for (int ii = 0; ii < MAX_A_SIZE_ / g.get_local_range(0); ii++) {
        int id  = ii * g.get_local_range(0) + tx;
        s_z_[id] = T(0);
        s_a_[id] = (id < num_a) ? d_a[id] : T(0);
      }
      it.barrier();

      for (int i = 0; i < yinfo_.dims[0]; i++) {
        if (tx == 0) {
          s_y_[0]    = __div((d_c[i] + s_z_[0]), s_a_[0]);
          d_y[i] = s_y_[0];
        }
        it.barrier();

        for (int ii = 0; ii < repeat; ii++) {
          int id = ii * g.get_local_range(0) + tx + 1;

          T z = s_z_[id] - __mul(s_a_[id], s_y_[0]);
          it.barrier();

          s_z_[id - 1] = z;
          it.barrier();
        }
      }
  }

private:
  write_accessor<T> yptr_;
  const KParam yinfo_;
  read_accessor<T> cptr_;
  const KParam cinfo_;
  read_accessor<T> aptr_;
  const KParam ainfo_;
  const int groups_y_;
local_accessor<T> s_z_;
local_accessor<T> s_a_;
local_accessor<T> s_y_;
const size_t MAX_A_SIZE_;
};


template<typename T, bool batch_a>
void iir(Param<T> y, Param<T> c, Param<T> a) {
    // FIXME: This is a temporary fix. Ideally the local memory should be
    // allocted outside
    constexpr int MAX_A_SIZE = (1024 * sizeof(double)) / sizeof(T);

    const int groups_y = y.info.dims[1];
    const int groups_x = y.info.dims[2];

    int threads = 256;
    while (threads > (int)y.info.dims[0] && threads > 32) threads /= 2;

    auto local = sycl::range(threads, 1);
    auto global = sycl::range(groups_x * local[0],
                       groups_y * y.info.dims[3] * local[1]);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_c{*c.data, h, sycl::read_only};
        sycl::accessor d_a{*a.data, h, sycl::read_only};
        sycl::accessor d_y{*y.data, h, sycl::write_only, sycl::no_init};
        local_accessor<T> s_z(MAX_A_SIZE, h);
        local_accessor<T> s_a(MAX_A_SIZE, h);
        local_accessor<T> s_y(1, h);
        h.parallel_for(
                       sycl::nd_range{global, local},
                       iirCreateKernel<T>(d_y, y.info, d_c, c.info, d_a, a.info, groups_y,
                                          s_z, s_a, s_y, MAX_A_SIZE));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
