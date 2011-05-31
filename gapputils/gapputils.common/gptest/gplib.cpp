// gptest.cpp : Defines the entry point for the console application.
//
#include "gplib.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <ostream>
#include <cmath>
#include <vector>
#include <cublas.h>
#include <cuda_runtime.h>
#include <cmath>

#include "chol.h"
#include "../optlib/ConjugateGradientsOptimizer.h"
#include "../optlib/OptimizerException.h"
#include "../optlib/GridSamplingOptimizer.h"
#include "NLML.h"

#include <blas/strsv.h>
#include <blas/sgemv.h>
#include <blas/strsm.h>
#include <blas/sgemm.h>

using namespace std;
using namespace optlib;

/// Row major matrix (as typical C style)
template<class T>
class Matrix {
private:
  T* data;
  int rowCount, columnCount;

public:
  Matrix(int rowCount, int columnCount) : rowCount(rowCount), columnCount(columnCount) {
    data = new T[rowCount * columnCount];
  }

  Matrix (T data[], int rowCount, int columnCount) : rowCount(rowCount), columnCount(columnCount) {
    this->data = new T[rowCount * columnCount];
    copy(data, data + (rowCount * columnCount), this->data);
  }

  virtual ~Matrix() {
    delete data;
  }

  T* getData() const {
    return data;
  }

  int getRowCount() const {
    return rowCount;
  }

  int getColumnCount() const {
    return columnCount;
  }

  bool isQuadratic() const {
    return getRowCount() == getColumnCount();
  }

  T& getElement(int rowIndex, int columnIndex) {
    return data[rowIndex * getColumnCount() + columnIndex];
  }

  T& operator()(int rowIndex, int columnIndex) {
    return getElement(rowIndex, columnIndex);
  }
};

template<class T>
ostream& operator<<(ostream& stream, const Matrix<T>& matrix) {
  const int rows = matrix.getRowCount();
  const int cols = matrix.getColumnCount();
  const T* data = matrix.getData();
  
  stream << "[ ";
  for (int i = 0, y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++i) {
      stream << data[i];
      if (x < cols - 1)
        stream << " ";
    }
    if (y < rows - 1)
      stream << "; ";
  }
  stream << " ]";

  return stream;
}

template<class T>
void chol(Matrix<T>& matrix) {
  T* elem = matrix.getData();
  if (!matrix.isQuadratic())
    throw "error";

  const int n = matrix.getColumnCount();
  for (int i = 0; i < n; ++i) {
    matrix(i, i) = sqrt(matrix(i, i));
    for (int j = i + 1; j < n; ++j) {
      matrix(i, j) /= matrix(i, i);
    }
    for (int j = i + 1; j < n; ++j) {
      for (int k = i + 1; k <=j; ++k) {
        matrix(k, j) -= matrix(i, j) * matrix(i, k);
      }
    }
  }

  // set lower triangular matrix to 0
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j)
      matrix(i, j) = 0;
}

namespace gplib {

void printMatrix(ostream& stream, const char* name, float *d_m, int m, int n, int pitch) {
  Matrix<float> M(n, m);

  //cublasGetVector(m * n, sizeof(float), d_m, 1, M.getData(), 1);
  cublasGetMatrix(m, n, sizeof(float), d_m, pitch, M.getData(), m);
  stream << name << " = " << M << "';" << endl;
}

GP* GP::instance = 0;

GP::GP() {
}

GP::~GP() {
}

GP& GP::getInstance() {
  if (!instance)
    instance = new GP();
  return *instance;
}

#define PRINT_MATRIX

void GP::gp(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length) {
#ifdef PRINT_MATRIX
  ofstream outfile("gpgpu.txt");
#endif

  int bn = n;
  if (n % 16) {
    bn += 16 - (n % 16);
  }

  float *d_K, *d_alpha, *d_mu, *d_Kstar, *d_x, *d_xstar, *d_length;
  cublasAlloc(bn * bn, sizeof(float), (void**)&d_K);
  cublasAlloc(n, sizeof(float), (void**)&d_alpha);
  cublasAlloc(m, sizeof(float), (void**)&d_mu);
  cublasAlloc(m * n, sizeof(float), (void**)&d_Kstar);
  cublasAlloc(n * d, sizeof(float), (void**)&d_x);
  cublasAlloc(m * d, sizeof(float), (void**)&d_xstar);
  cublasAlloc(d, sizeof(float), (void**)&d_length);

  cublasSetVector(n, sizeof(float), y, 1, d_alpha, 1);
  cublasSetVector(n * d, sizeof(float), x, 1, d_x, 1);
  cublasSetVector(m * d, sizeof(float), xstar, 1, d_xstar, 1);
  cublasSetVector(d, sizeof(float), length, 1, d_length, 1);

#ifdef PRINT_MATRIX
  printMatrix(outfile, "x", d_x, n, d, n);
  printMatrix(outfile, "xstar", d_xstar, m, d, m);
#endif
  covSE(n, n, d, d_K, bn, d_x, n, d_x, n, sigmaF, sigmaN, d_length);

  // calculated in columne major equals calculate Kstar' in row major
  covSE(m, n, d, d_Kstar, n, d_xstar, m, d_x, n, sigmaF, 0.0, d_length);
#ifdef PRINT_MATRIX
  printMatrix(outfile, "K", d_K, n, n, bn);
#endif
  cholesky_cuda_block(d_K, bn);
#ifdef PRINT_MATRIX
  printMatrix(outfile, "L", d_K, n, n, bn);
#endif
  cublasStrsv('U', 'T', 'N', n, d_K, bn, d_alpha, 1);
  cublasStrsv('U', 'N', 'N', n, d_K, bn, d_alpha, 1);

  cublasSgemv('T', n, m, 1.0, d_Kstar, n, d_alpha, 1, 0.0, d_mu, 1);
  cublasGetVector(m, sizeof(float), d_mu, 1, mu, 1);

  if (cov) {
    float *d_Kss;

    cublasAlloc(m * m, sizeof(float), (void**)&d_Kss);
    covSE(m, m, d, d_Kss, m, d_xstar, m, d_xstar, m, sigmaF, 0.0, d_length);
#ifdef PRINT_MATRIX
    printMatrix(outfile, "L", d_K, n, n, bn);
    printMatrix(outfile, "Kstar", d_Kstar, n, m, n);
    printMatrix(outfile, "Kss", d_Kss, m, m, m);
#endif
    // v = L\Kstar
    cublasStrsm('L', 'U', 'T', 'N', n, m, 1.f, d_K, bn, d_Kstar, n);
#ifdef PRINT_MATRIX
    printMatrix(outfile, "V", d_Kstar, n, m, n);
#endif
    // cov = Kss - V'*V
    cublasSgemm('t', 'n', m, m, n, -1, d_Kstar, n, d_Kstar, n, 1.0, d_Kss, m);
#ifdef PRINT_MATRIX
    printMatrix(outfile, "cov", d_Kss, m, m, m);
#endif
    cublasGetVector(m * m, sizeof(float), d_Kss, 1, cov, 1);

    cublasFree(d_Kss);
  }

  cublasFree(d_K);
  cublasFree(d_alpha);
  cublasFree(d_mu);
  cublasFree(d_Kstar);
  cublasFree(d_x);
  cublasFree(d_xstar);
  cublasFree(d_length);

#ifdef PRINT_MATRIX
  outfile.close();
#endif
}

void printMatrixCPU(ostream& stream, const char* name, float *h_m, int m, int n) {
  Matrix<float> M(n, m);

  copy(h_m, h_m + (m * n), M.getData());
  stream << name << " = " << M << "';" << endl;
}

float se(float* x1, int ldx1, float* x2, int ldx2, float s2, float* l, int d) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float diff = x1[i * ldx1] - x2[i * ldx2];
    sum += diff * diff / (l[i] * l[i]);
  }
  return s2 * exp(-0.5 * sum);
}

void covSECpu(int m, int n, int d, float *K, float *U, float *V, float sigmaF, float sigmaN, float* length) {
  for (int j = 0, k = 0; j < n; ++j)
    for (int i = 0; i < m; ++i, ++k)
      K[k] = se(&U[i], n, &V[j], n, sigmaN * sigmaN, length, d) + (i == j ? sigmaF * sigmaF : 0);
}

GPTEST_API void gpcpu(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length) {
  const float s2 = sigmaF * sigmaF;
  const float s22 = sigmaN * sigmaN;

#ifdef PRINT_MATRIX
  ofstream outfile("gpcpu.txt");
#endif

  float *K = new float[n*n];
  float *Kstar = new float[m * n];

  /*for (int i = 0, k = 0; i < n; ++i)
    for (int j = 0; j < n; ++j, ++k)
      K[k] = se(&x[i], n, &x[j], n, s2, length, d) + (i == j ? s22 : 0);
      */
  covSECpu(n, n, d, K, x, x, sigmaF, sigmaN, length);

  // compute Kstar in column major order
  /*for (int j = 0, k = 0; j < m; ++j)
    for (int i = 0; i <n; ++i, ++k)
      Kstar[k] = se(&x[i], n, &xstar[j], m, s2, length, d);
  */
  covSECpu(m, n, d, Kstar, xstar, x, sigmaF, 0.0, length);

  // Calculate cholesky decomposition L
  vector<float> alpha(y, y + n);

  Matrix<float> L(K, n, n);
  chol(L);

#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "x", x, n, d);
  printMatrixCPU(outfile, "xstar", xstar, m, d);
  printMatrixCPU(outfile, "K", K, n, n);
  printMatrixCPU(outfile, "L", L.getData(), n, n);
  printMatrixCPU(outfile, "Kstar", Kstar, n, m);
  printMatrixCPU(outfile, "y", &alpha[0], n, 1);
#endif
  integer ldalpha = 1;
  integer _n = n;
  integer _m = m;
  strsv_("L", "N", "N", &_n, L.getData(), &_n, &alpha[0], &ldalpha);
#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "alpha", &alpha[0], n, 1);
#endif
  strsv_("L", "T", "N", &_n, L.getData(), &_n, &alpha[0], &ldalpha);
#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "alpha", &alpha[0], n, 1);
#endif
  real one = 1.0;
  real zero = 0.0;
  sgemv_("T", &_n, &_m, &one, Kstar, &_n, &alpha[0], &ldalpha, &zero, mu, &ldalpha);
#ifdef PRINT_MATRIX  
  printMatrixCPU(outfile, "mu", mu, m, 1);
#endif

  if (cov) {
    float *Kss = new float[m * m];

    for (int i = 0, k = 0; i < m; ++i)
      for (int j = 0; j < m; ++j, ++k)
        Kss[k] = se(&xstar[i], m, &xstar[j], m, s2, length, d);

#ifdef PRINT_MATRIX
    printMatrixCPU(outfile, "Kss", Kss, m, m);
    printMatrixCPU(outfile, "L", L.getData(), n, n);
    printMatrixCPU(outfile, "Kstar", Kstar, n, m);
#endif
    // v = L\Kstar
    strsm_("L", "L", "N", "N", &_n, &_m, &one, L.getData(), &_n, Kstar, &_n);
#ifdef PRINT_MATRIX
    printMatrixCPU(outfile, "V", Kstar, n, m);
#endif
    // cov = Kss - V'*V
    real minusOne = -1.0;
    sgemm_("t", "n", &_m, &_m, &_n, &minusOne, Kstar, &_n, Kstar, &_n, &one, Kss, &_m);
#ifdef PRINT_MATRIX
    printMatrixCPU(outfile, "cov", Kss, m, m);
#endif
    copy(Kss, Kss + m * m, cov);
    delete Kss;
  }

  delete K;
  delete Kstar;
#ifdef PRINT_MATRIX
  outfile.close();
#endif
}

void GP::trainGP(float& sigmaF, float& sigmaN, float* length, float* x, float* y, int n, int d) {
  NLML nlml(x, y, n, d);

  vector<double> params(2 + d);
  for (int i = 0; i < params.size(); ++i)
    params[i] = 0;

  GridParameter gridParams;

  // Sigma f
  vector<double> sigmaFDir(3);
  sigmaFDir[0] = 1.0;
  sigmaFDir[1] = 0.0;
  sigmaFDir[2] = 0.0;

  GridLineParameter sigmaFLine;
  sigmaFLine.direction = sigmaFDir;
  sigmaFLine.minValue = -3;
  sigmaFLine.maxValue = 3;
  sigmaFLine.samplesCount = 3;
  gridParams.gridLines.push_back(sigmaFLine);

  // Sigma n
  vector<double> sigmaNDir(3);
  sigmaNDir[0] = 0.0;
  sigmaNDir[1] = 1.0;
  sigmaNDir[2] = 0.0;

  GridLineParameter sigmaNLine;
  sigmaNLine.direction = sigmaNDir;
  sigmaNLine.minValue = -3;
  sigmaNLine.maxValue = 3;
  sigmaNLine.samplesCount = 5;
  gridParams.gridLines.push_back(sigmaNLine);

  // Length
  vector<double> lengthDir(2 + d);
  lengthDir[0] = 0.0;
  lengthDir[1] = 0.0;
  for (int i = 0; i < d; ++i)
    lengthDir[i + 2] = 1.0;

  GridLineParameter lengthLine;
  lengthLine.direction = lengthDir;
  lengthLine.minValue = -3;
  lengthLine.maxValue = 3;
  lengthLine.samplesCount = 5;
  gridParams.gridLines.push_back(lengthLine);

  GridSamplingOptimizer<ConjugateGradientsOptimizer> optimizer;
  optimizer.setParameter(Parameter::GridSampling, &gridParams);
  try {
    optimizer.minimize(params, nlml);
  } catch (optlib::OptimizerException ex) {
    cout << ex.what() << endl;
  }

  sigmaF = exp(params[0]);
  sigmaN = exp(params[1]);
  for (int i = 0; i < d; ++i)
    length[i] = exp(params[i + 2]);
}

}
