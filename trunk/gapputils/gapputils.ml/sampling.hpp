/*
 * sampling.hpp
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_SAMPLING_HPP_
#define GAPPUTILS_ML_SAMPLING_HPP_

#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

namespace gapputils {

namespace ml {

template<class T>
struct get_randn : public thrust::unary_function<unsigned int, T> {
  T mean, stddev;

  get_randn(const T& mean, const T& stddev) : mean(mean), stddev(stddev) { }

  __host__ __device__
  unsigned int hash(unsigned int a) const
  {
      a = (a+0x7ed55d16) + (a<<12);
      a = (a^0xc761c23c) ^ (a>>19);
      a = (a+0x165667b1) + (a<<5);
      a = (a+0xd3a2646c) ^ (a<<9);
      a = (a+0xfd7046c5) + (a<<3);
      a = (a^0xb55a4f09) ^ (a>>16);
      return a;
  }

  __host__ __device__ T operator()(unsigned i) const {
    thrust::default_random_engine rng(hash(i));
    thrust::random::experimental::normal_distribution<T> dist(mean, stddev);

    return dist(rng);
  }
};

template<class T>
struct sample_units : public thrust::binary_function<T, unsigned int, T> {

  __host__ __device__
  unsigned int hash(unsigned int a) const
  {
      a = (a+0x7ed55d16) + (a<<12);
      a = (a^0xc761c23c) ^ (a>>19);
      a = (a+0x165667b1) + (a<<5);
      a = (a+0xd3a2646c) ^ (a<<9);
      a = (a+0xfd7046c5) + (a<<3);
      a = (a^0xb55a4f09) ^ (a>>16);
      return a;
  }

  __host__ __device__ T operator()(const T& x, unsigned i) const {
    thrust::default_random_engine rng(hash(i));
    thrust::uniform_real_distribution<T> u01(0,1);

    return x > u01(rng);
  }
};

}

}

#endif /* GAPPUTILS_ML_SAMPLING_HPP_ */
