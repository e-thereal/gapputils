#include "NlmlCpu.h"

#include <algorithm>
#include <fstream>
#include <cmath>

#include <blas/strsv.h>

using namespace std;

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

  

void printMatrixCPU(ostream& stream, const char* name, float *h_m, int m, int n);
void covSECpu(int m, int n, int d, float *K, float *U, float *V, float sigmaF, float sigmaN, float* length);

NlmlCpu::NlmlCpu(float* x, float *y, int n, int d) : n(n), d(d), K(n, n), alpha(n), diag(n),
  x(x, x + n * d), y(y, y + n)
{
}


NlmlCpu::~NlmlCpu(void)
{
}

#define PRINT_MATRIX

double NlmlCpu::eval(float sigmaF, float sigmaN, float* length)  {
  const float logTwoPiHalf = 0.918938533f;
#ifdef PRINT_MATRIX
  std::ofstream outfile("nlmlcpu.txt");
#endif
  copy(y.begin(), y.end(), alpha.begin());

  covSECpu(n, n, d, K.getData(), &x[0], &x[0], sigmaF, sigmaN, length);
#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "K", K.getData(), n, n);
  printMatrixCPU(outfile, "x", &x[0], n, d);
  printMatrixCPU(outfile, "y", &alpha[0], n, 1);
#endif
  chol(K);
#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "L", K.getData(), n, n);
#endif
  integer _n = n;
  integer _lda = 1;
  strsv_("L", "N", "N", &_n, K.getData(), &_n, &alpha[0], &_lda);
#ifdef PRINT_MATRIX
  printMatrixCPU(outfile, "alpha", &alpha[0], n, 1);
#endif
  // 0.5 * alpha' * alpha + log(det(K)) + n/2 * ln(2 PI)
  // Inner product
  float inner = 0.0;
  for (int i = 0; i < n; ++i)
    inner += alpha[i] * alpha[i];

  // Determinate
  float logdet = 0.0;
  for (int i = 0; i < n; ++i)
    logdet += log(K(i, i));

  float ret = 0.5f * inner + logdet + n * logTwoPiHalf;
#ifdef PRINT_MATRIX
  outfile << "inner = " << inner << endl;
  outfile << "logdetL = " << logdet << endl;
  outfile << "nlml = " << ret << endl;
  outfile.close();
#endif
  return ret;
}

}
