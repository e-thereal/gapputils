#ifndef CHOL_H_
#define CHOL_H_

#include "StdAfx.h"

#define CUDA_MAX_BLOCK_SIZE 32
#define BLOCK_SIZE 16

#define DT float

void cholesky_cuda(DT *d_m, int n);
/// n must be a multiply of 16 (fixed block size)
void cholesky_cuda_block(DT *d_m, int n, int bsize = 16);

// d_diag = log(diag(d_m)) : d_m is a n x n matrix
void logdiag(DT *d_diag, DT *d_m, int n, int pitch);

// calculated the determinante of a triangular matrix
DT Strldet(DT *d_m, int n, int pitch);

DT l2norm(DT *d_v, int n);

// K is m x n, U is d x m and V is d x n, all matrices are in row major order, leading dimension (ld) is number of columns
void covSE(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length);
void covSEFast(int m, int n, int d, DT *d_K, int ldk, DT *d_U, int ldu, DT *d_V, int ldv, DT sigmaF, DT sigmaN, DT* d_length);

#endif
