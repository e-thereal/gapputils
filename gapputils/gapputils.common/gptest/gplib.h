#pragma once

#ifndef _GPLIB_GP_H
#define _GPLIB_GP_H

#include <vector>
#include <ostream>

#ifdef GPTEST_EXPORT
#define GPTEST_API __declspec(dllexport)
#else
#define GPTEST_API __declspec(dllimport)
#pragma comment (lib, "gptest")
#endif

namespace gplib {

void printMatrix(std::ostream& stream, const char* name, float *d_m, int m, int n, int pitch);

class GPTEST_API GP {
private:
  static GP* instance;
  
protected:
  GP();

public:
  virtual ~GP();

  static GP& getInstance();

  // mu must be size of xstar
  void gp(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length);

  // the search is always reset to 1, 1, 1
  void trainGP(float& sigmaF, float& sigmaN, float* length, float* x, float* y, int n, int d);
};

GPTEST_API void gpcpu(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length);

}

#endif