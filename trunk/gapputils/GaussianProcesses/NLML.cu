#include "NLML.h"

#include <thrust/copy.h>
#include <cassert>
#include <cublas.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <vector>

#include "gpgpu.h"

using namespace std;

namespace GaussianProcesses {

NLML::NLML(float* x, float *y, int n, int d)
 : n(n), d(d), d_alpha(n), d_diag(n),
   d_x(x, x + (n * d)), d_y(y, y + n), d_length(d), length(d)
{
  bn = n;
  if (n % BLOCK_SIZE) {
    bn += BLOCK_SIZE - (n % BLOCK_SIZE);
  }
  d_K.resize(bn * bn);
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

NLML::DomainType NLML::gradient(const DomainType& parameter) {
  assert(parameter.size() == 2 + d);
  const float sigmaF = exp(parameter[0]);
  const float sigmaN = exp(parameter[1]);

  for (int i = 0; i < d; ++i)
    length[i] = exp(parameter[i + 2]);

  return gradient(sigmaF, sigmaN, &length[0]);
}

#define PRINT_MATRIX

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

NLML::DomainType NLML::gradient(float sigmaF, float sigmaN, float* length) {
  vector<double> gradient(d + 2);
  ofstream outfile("dnlml.txt");

  thrust::copy(d_y.begin(), d_y.end(), d_alpha.begin());
  printMatrix(outfile, "y", d_alpha.data().get(), n, 1, n);
  thrust::copy(length, length + d, d_length.begin());
  covSEFast(n, n, d, d_K.data().get(), bn, d_x.data().get(), n, d_x.data().get(), n, sigmaF, sigmaN, d_length.data().get());
  printMatrix(outfile, "K", d_K.data().get(), n, n, bn);

  cholesky_cuda_block(d_K.data().get(), bn);
  printMatrix(outfile, "L", d_K.data().get(), n, n, bn);

  // alpha = L'\(L\y) = K^{-1}y
  cublasStrsv('U', 'T', 'N', n, d_K.data().get(), bn, d_alpha.data().get(), 1);
  cublasStrsv('U', 'N', 'N', n, d_K.data().get(), bn, d_alpha.data().get(), 1);
  printMatrix(outfile, "alpha", d_alpha.data().get(), n, 1, n);

  // d_W = invK = L'\(L\I) (I = identity)
  thrust::device_vector<float> d_W(n * n);
  setToIdentity(d_W.data().get(), n);
  printMatrix(outfile, "I", d_W.data().get(), n, n, n);
  cublasStrsm('L', 'U', 'T', 'N', n, n, 1.f, d_K.data().get(), bn, d_W.data().get(), n);
  cublasStrsm('L', 'U', 'N', 'N', n, n, 1.f, d_K.data().get(), bn, d_W.data().get(), n);
  printMatrix(outfile, "invK", d_W.data().get(), n, n, n);

  // W = invK - alpha * alpha' = -1.0 * alpha * alpha' + 1.0 * invK
  cublasSgemm('N', 'T', n, n, 1, -1.f, d_alpha.data().get(), n, d_alpha.data().get(), n,
      1.0f, d_W.data().get(), n);
  printMatrix(outfile, "W", d_W.data().get(), n, n, n);

  thrust::device_vector<float> d_dK(n * n);
  for (int iparam = 0; iparam < gradient.size(); ++iparam) {
    stringstream sname;
    sname << "dK_" << iparam;

    // dK/dO_i
    derivSE(n, n, d, d_dK.data().get(), n, d_x.data().get(), n, d_x.data().get(), n, sigmaF, sigmaN, d_length.data().get(), iparam);
    printMatrix(outfile, sname.str().c_str(), d_dK.data().get(), n, n, n);

    // dnlml_i = 1/2 * tr(W * dK/dO_i) = 1/2 * sum(W .* dK/dO_i)
    gradient[iparam] = 0.5 * thrust::inner_product(d_W.begin(), d_W.end(), d_dK.begin(), 0.0f);

    outfile << "g_" << iparam << " = " << gradient[iparam] << ";" << endl;
  }
  outfile.close();

  return gradient;
}

}
