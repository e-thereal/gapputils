#pragma once

#ifndef _GPLIB_NLML_H_
#define _GPLIB_NLML_H_

#include "../optlib/IMultidimensionOptimizer.h"
#include <thrust/device_vector.h>
#include <vector>

#include "gplib.h"

namespace gplib {

class GPTEST_API NLML : public virtual optlib::IFunction<optlib::IMultidimensionOptimizer::DomainType>
{
private:
  int n, bn, d;
  thrust::device_vector<float> d_K, d_alpha, d_diag, d_x, d_y, d_length;
  std::vector<float> length;

public:
  NLML(float* x, float *y, int n, int d);
  ~NLML(void);

  virtual double eval(const DomainType& parameter);
  double eval(float sigmaF, float sigmaN, float* length);
};

}

#endif
