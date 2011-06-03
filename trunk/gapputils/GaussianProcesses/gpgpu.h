/*
 * gpgpu.h
 *
 *  Created on: Jun 3, 2011
 *      Author: tombr
 */

#ifndef GAUSSIANPROCESS_GPGPU_H_
#define GAUSSIANPROCESS_GPGPU_H_

#define CUDA_MAX_BLOCK_SIZE 16
#define BLOCK_SIZE 16
#define DT float

namespace GaussianProcesses {

void gp(float* mu, float* cov, float* x, float* y, float* xstar, int n, int d, int m, float sigmaF, float sigmaN, float* length);

// the search is always reset to 1, 1, 1
void trainGP(float& sigmaF, float& sigmaN, float* length, float* x, float* y, int n, int d);

/*** internal cuda functions ***/

void cholesky_cuda(DT *d_m, int n);

/// n must be a multiply of 16 (fixed block size)
void cholesky_cuda_block(DT *d_m, int n, int bsize = 16);

DT l2norm(DT *d_v, int n);
DT Strldet(DT *d_m, int n, int pitch);

// K is m x n, U is d x m and V is d x n, all matrices are in row major order, leading dimension (ld) is number of columns
void covSE(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length);
void covSEFast(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length);

}

#endif /* GPGPU_H_ */
