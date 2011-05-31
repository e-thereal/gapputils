#include "chol.h"

#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

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

  return reduce(d_diag.begin(), d_diag.end(), 1.0, thrust::plus<DT>());
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

void covSE(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT *d_length) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  covSEKernel<<<blocks, threads>>>(m, n, d, d_K, ldk, d_U, ldu, d_V, ldv, sigmaF, sigmaN, d_length);
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

void covSEFast(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT *d_length) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  covSEKernelFast<<<blocks, threads>>>(m, n, d, d_K, ldk, d_U, ldu, d_V, ldv, sigmaF, sigmaN, d_length);
  cudaThreadSynchronize();
}
