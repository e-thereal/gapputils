/*
 * gpgpu.cu
 *
 *  Created on: Jun 3, 2011
 *      Author: tombr
 */

#include "gpgpu.h"

#include <cublas.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <cuda_runtime.h>

using namespace std;

namespace GaussianProcesses {

// cublas causes a segfault on linux. It's a known bug without a good fix.
void gp(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length) {
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

  covSE(n, n, d, d_K, bn, d_x, n, d_x, n, sigmaF, sigmaN, d_length);

  // calculated in column major equals calculate Kstar' in row major
  covSE(m, n, d, d_Kstar, n, d_xstar, m, d_x, n, sigmaF, 0.0, d_length);
  cholesky_cuda_block(d_K, bn);
  cublasStrsv('U', 'T', 'N', n, d_K, bn, d_alpha, 1);
  cublasStrsv('U', 'N', 'N', n, d_K, bn, d_alpha, 1);

  cublasSgemv('T', n, m, 1.0, d_Kstar, n, d_alpha, 1, 0.0, d_mu, 1);
  cublasGetVector(m, sizeof(float), d_mu, 1, mu, 1);

  if (cov) {
    float *d_Kss;

    cublasAlloc(m * m, sizeof(float), (void**)&d_Kss);
    covSE(m, m, d, d_Kss, m, d_xstar, m, d_xstar, m, sigmaF, 0.0, d_length);

    // v = L\Kstar
    cublasStrsm('L', 'U', 'T', 'N', n, m, 1.f, d_K, bn, d_Kstar, n);

    // cov = Kss - V'*V
    cublasSgemm('t', 'n', m, m, n, -1, d_Kstar, n, d_Kstar, n, 1.0, d_Kss, m);
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
}



/*** INTERNAL STUFF ***/

__global__ void cuda_chol_iter(DT *m, int n, int boffset) {
  int k;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bsize = blockDim.x;

  __shared__ DT b[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];

  b[x][y] = m [(y + boffset) * n + boffset + x];

  for (k = 0; k < bsize; ++k) {
    __syncthreads();
    if (x == k) {
      DT fac = sqrtf(b[x][x]);
      if (y >= x)
        b[x][y] /= fac;
    }
    __syncthreads();
    if (x > k && y >= x) {
      b[x][y] -= b[k][y] * b[k][x];
    }
  }
  __syncthreads();
  m[(boffset + y) * n + boffset + x] = b[x][y];
}

void cholesky_cuda(DT *d_m, int n) {
  assert(n <= 16);
  dim3 threads(n, n);
  cuda_chol_iter <<<1, threads>>>(d_m, n, 0);
  cudaThreadSynchronize();
}

__global__ void cuda_strip_block(DT *m, int n, int boffset) {
  int k;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bsize = blockDim.x;
  int by = (blockIdx.y+1) * bsize + boffset;

  __shared__ DT b[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];
  __shared__ DT ttl[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];
  __shared__ DT rttl[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];

  b[x][y] = m[(by + y) * n + boffset + x];
  ttl[y][x] = m[(boffset + y) * n + boffset + x];
  __syncthreads();

  if (y == 0) {
    int k;
    for (k = bsize - 1; k >= 0; --k) {
      int m;
      DT dotprod = (k == x ? (DT)1.0 : (DT)0.0);
      for (m = bsize - 1; m > k; --m) {
        dotprod -= ttl[m][k] * rttl[x][m];
      }
      rttl[x][k] = dotprod / ttl[k][k];
    }
  }
  __syncthreads();

  DT a = 0;
  for (k = 0; k < bsize; ++k) {
    a += b[k][y] * rttl[x][k];
  }

  m[(by + y) * n + boffset + x] = a;
}

__global__ void cuda_lower_right(DT *m, int n, int boffset) {
  int k;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bsize = blockDim.x;
  int bx = boffset + (blockIdx.x + 1) * bsize;
  int by = boffset + (blockIdx.y + 1) * bsize;

  if (by < bx)
    return;

  __shared__ DT a[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];
  __shared__ DT b[CUDA_MAX_BLOCK_SIZE][CUDA_MAX_BLOCK_SIZE];

  a[x][y] = m[(by + y) * n + boffset + x];
  b[x][y] = m[(bx + x) * n + boffset + y];
  __syncthreads();

  DT u = 0;
  for (k = 0; k < bsize; ++k) {
    u += a[k][y] * b[x][k];
  }

  m[(by + y) * n + bx + x] -= u;
}

void cholesky_cuda_block(DT *d_m, int n, int bsize) {
  assert(n % bsize == 0);
  //int size = n * n * sizeof(DT);
  int iter = (n + bsize - 1) / bsize;
  int i;
  dim3 threads(bsize, bsize);

  for (i = 0; i < iter; ++i) {
    // top-left block
    cuda_chol_iter<<<1, threads>>>(d_m, n, i * bsize);
    cudaThreadSynchronize();

    // Left strip
    if (iter - i - 1 > 0) {
      dim3 strip(1, iter - i - 1, 1);
      cuda_strip_block<<<strip, threads>>>(d_m, n, i * bsize);
      cudaThreadSynchronize();
    }

    // Lower-right matrix
    if (iter - i - 1 > 0) {
      dim3 lr(iter - i - 1, iter -i - 1, 1);
      cuda_lower_right<<<lr, threads>>>(d_m, n, i * bsize);
      cudaThreadSynchronize();
    }
  }
}

__global__ void logdiagKernel(DT *d_diag, DT *d_m, int n, int pitch) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n)
    d_diag[idx] = logf(d_m[idx + idx * pitch]);
}

void logdiag(DT *d_diag, DT *d_m, int n, int pitch) {
  dim3 blockSize(256);
  dim3 gridSize((n + 255) / 256);

  logdiagKernel<<<blockSize, gridSize>>>(d_diag, d_m, n, pitch);
  cudaThreadSynchronize();
}

DT Strldet(DT *d_m, int n, int pitch) {
  using namespace thrust;

  device_vector<DT> d_diag(n);
  logdiag(d_diag.data().get(), d_m, n, pitch);

  return reduce(d_diag.begin(), d_diag.end(), 0.0f, thrust::plus<DT>());
}

DT l2norm(DT *d_v, int n) {
  thrust::device_ptr<float> d_vptr(d_v);
  return thrust::inner_product(d_vptr, d_vptr + n, d_vptr, 0.0f);
}

__global__ void covSEKernel(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= m || j >= n)
    return;

  //printf("i: %d\n", i);
  //printf("j: %d\n", j);
  //printf("d: %d\n", d);
  //printf("m: %d\n", m);
  //printf("n: %d\n", n);
  //printf("d_U: %f\n", d_U[i]);
  //printf("d_V: %f\n", d_V[j]);
  //printf("l: %f\n", d_length[0]);

  DT sum = 0;
  for (int k = 0; k < d; ++k) {
    float diff = d_U[i + k * ldu] - d_V[j + k * ldv];
    //printf("diff: %f\n", diff);
    sum += diff * diff / (d_length[k] * d_length[k]);
  }

  //printf("sF: %f\n", sigmaF);
  //printf("sN: %f\n", sigmaN);
  //printf("sum: %f\n", sum);
  //printf("exp: %f\n", expf(-0.5 * sum));
  //d_K[j + i * ldk] = d_U[i + k * ldu];
  d_K[j + i * ldk] = sigmaF * sigmaF * expf(-1.f/2.f * sum) + (i == j ? sigmaN * sigmaN : 0);
  //printf("ldk: %d\n", ldk);
  //printf("d_K: %f\n", d_K[j + i * ldk]);
}

// K is m x n, U is d x m and V is d x n, all matrices are in row major order, leading dimension (ld) is number of columns
void covSE(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  covSEKernel<<<blocks, threads>>>(m, n, d, d_K, ldk, d_U, ldu, d_V, ldv, sigmaF, sigmaN, d_length);
  cudaThreadSynchronize();
}

__global__ void setToIdentityKernel(DT *d_M, int n) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= n || j >= n)
    return;

  d_M[j + i * n] = (i == j);
}

void setToIdentity(DT *d_M, int n) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  setToIdentityKernel<<<blocks, threads>>>(d_M, n);
  cudaThreadSynchronize();
}

__global__ void covSEKernelFast(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  __shared__ DT s_U[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ DT s_V[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ DT s_length[BLOCK_SIZE];

  DT sum = 0;

  for (int b = 0; b < d; b += BLOCK_SIZE) {
    // load next bunch into shared memory
    if (i < m && (b + threadIdx.x) < d)
      s_U[threadIdx.y][threadIdx.x] = d_U[i + (b + threadIdx.x) * ldu];

    if (j < n && (b + threadIdx.y) < d)
      s_V[threadIdx.x][threadIdx.y] = d_V[j + (b + threadIdx.y) * ldv];

    if (threadIdx.y == 0 && (b + threadIdx.x) < d)
      s_length[threadIdx.x] = d_length[b + threadIdx.x];

    __syncthreads();

    if (i < m && j < n) {
      // Process loaded data
      for (int k = 0; k < BLOCK_SIZE && b + k < d; ++k) {
        float diff = s_U[threadIdx.y][k] - s_V[threadIdx.x][k];
        sum += diff * diff / (s_length[k] * s_length[k]);
      }
    }
    __syncthreads();
  }

  if (i < m && j < n)
    d_K[j + i * ldk] = sigmaF * sigmaF * expf(-0.5f * sum) + (i == j ? sigmaN * sigmaN : 0);
}

void covSEFast(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  covSEKernelFast<<<blocks, threads>>>(m, n, d, d_K, ldk, d_U, ldu, d_V, ldv, sigmaF, sigmaN, d_length);
  cudaThreadSynchronize();
}

__global__ void derivSEKernel(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu,
    DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length, int param)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= m || j >= n)
    return;

  switch (param) {
  case 0: {
    // sigma f
    DT sum = 0;
    for (int k = 0; k < d; ++k) {
      float diff = (d_U[i + k * ldu] - d_V[j + k * ldv]) / d_length[k];
      sum += diff * diff;
    }

    d_K[j + i * ldk] = 2.f * expf(-1.f/2.f * sum);
  } break;

  case 1:
    // sigma n
    d_K[j + i * ldk] = 2.f * (i == j);
    break;

  default: {
      // length
      const int lparam = param - 2;
      DT sum = 0;
      for (int k = 0; k < d; ++k) {
        float diff = d_U[i + k * ldu] - d_V[j + k * ldv];
        sum += diff * diff / (d_length[k] * d_length[k]);
      }
      const float diff = d_U[i + lparam * ldu] - d_V[j + lparam * ldv];
      const float lplenght = d_length[lparam];

      d_K[j + i * ldk] = sigmaF * sigmaF * expf(-1.f/2.f * sum) * diff * diff / (lplenght * lplenght * lplenght);
    }
  }
}

void derivSE(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length, int param) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  derivSEKernel<<<blocks, threads>>>(m, n, d, d_K, ldk, d_U, ldu, d_V, ldv, sigmaF, sigmaN, d_length, param);
  cudaThreadSynchronize();
}

}
