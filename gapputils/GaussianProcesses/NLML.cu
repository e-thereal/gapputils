#include "NLML.h"

#include <thrust/copy.h>
#include <cassert>
#include <cublas.h>
#include <fstream>
#include <iostream>

#include <vector>

#include "gpgpu.h"

using namespace std;

namespace GaussianProcesses {

NLML::NLML(float* x, float *y, int n, int d) : n(n), d(d), d_alpha(n), d_diag(n),
  d_x(x, x + (n * d)), d_y(y, y + n), d_length(d), length(d)
{
  bn = n;
  if (n % BLOCK_SIZE) {
    bn += BLOCK_SIZE - (n % BLOCK_SIZE);
  }
  d_K.resize(bn * bn);
  vector<float> vx(n * d);
  thrust::copy(d_x.begin(), d_x.end(), vx.begin());
  ;
}

NLML::~NLML(void)
{
}

double NLML::eval(const DomainType& parameter) {

  assert(parameter.size() == 2 + d);
  const float sigmaF = exp(parameter[0]);
  const float sigmaN = exp(parameter[1]);
  
  for (int i = 0; i < d; ++i)
    length[i] = exp(parameter[i + 2]);

  return eval(sigmaF, sigmaN, &length[0]);
}

//#define PRINT_MATRIX

double NLML::eval(float sigmaF, float sigmaN, float* length) {
  const float logTwoPiHalf = 0.918938533f;
#ifdef PRINT_MATRIX
  std::ofstream outfile("nlmlgpu.txt");
#endif
  thrust::copy(d_y.begin(), d_y.end(), d_alpha.begin());
  thrust::copy(length, length + d, d_length.begin());
  covSEFast(n, n, d, d_K.data().get(), bn, d_x.data().get(), n, d_x.data().get(), n, sigmaF, sigmaN, d_length.data().get());
#ifdef PRINT_MATRIX  
  printMatrix(outfile, "Kfast", d_K.data().get(), n, n, bn);
#endif
  //covSE(n, n, d, d_K.data().get(), bn, d_x.data().get(), n, d_x.data().get(), n, sigmaF, sigmaN, d_length.data().get());
#ifdef PRINT_MATRIX
  printMatrix(outfile, "K", d_K.data().get(), n, n, bn);
  printMatrix(outfile, "x", d_x.data().get(), n, d, n);
  printMatrix(outfile, "y", d_alpha.data().get(), n, 1, n);
#endif
  cholesky_cuda_block(d_K.data().get(), bn);
#ifdef PRINT_MATRIX
  printMatrix(outfile, "U", d_K.data().get(), n, n, bn);
#endif
  cublasStrsv('U', 'T', 'N', n, d_K.data().get(), bn, d_alpha.data().get(), 1);
#ifdef PRINT_MATRIX
  printMatrix(outfile, "alpha", d_alpha.data().get(), n, 1, n);
#endif
  // 0.5 * alpha' * alpha + log(det(K)) + n/2 * ln(2 PI)
  float l2 = l2norm(d_alpha.data().get(), n);
  float logDet = Strldet(d_K.data().get(), n, bn);
  //cout << "L2: " << l2 << endl;
  //cout << "logDet: " << logDet << endl;

  float ret = 0.5f * l2 + logDet + n * logTwoPiHalf;
#ifdef PRINT_MATRIX
  outfile << "nlml = " << ret << std::endl;
  outfile.close();
#endif
  return ret;
}

}
