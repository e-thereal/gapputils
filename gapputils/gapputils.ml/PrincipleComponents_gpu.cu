/*
 * PrincipleComponents_gpu.cu
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include <cuda_runtime.h>
#include <cula_lapack_device.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <iostream>

#include <tbblas/device_vector.hpp>
#include <tbblas/device_matrix.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <algorithm>

namespace gapputils {

namespace ml {

void getPcs(float* pc, float* data, int m, int n) {
  using namespace thrust::placeholders;
  namespace ublas = boost::numeric::ublas;

  const int count = m * n;
  culaStatus status;

  thrust::device_vector<float> d_U(m * m);
  thrust::device_vector<float> d_sigma(n);
  thrust::device_vector<float> d_Vt(n * n);

  ublas::matrix<float> umat(n, m);
  std::copy(data, data + count, umat.data().begin());

  tbblas::device_matrix<float> mat(n, m);
  mat = umat;

  for (int iCol = 0; iCol < m; ++iCol) {
    tbblas::device_vector<float> column = tbblas::column(mat, iCol);
    column -= tbblas::sum(column) / column.size();
  }

  umat = mat;

  thrust::device_vector<float> d_data(umat.data().begin(), umat.data().end());

  //culaSgesvd
  char jobu = 'A';
  char jobvt = 'A';
  if ((status = culaDeviceSgesvd(jobu, jobvt, m, n, d_data.data().get(), m, d_sigma.data().get(),
      d_U.data().get(), m, d_Vt.data().get(), n)) != culaNoError)
  {
    std::cout << "Could not SVD: " << culaGetStatusString(status) << std::endl;
    return;
  }

  thrust::copy(d_U.begin(), d_U.end(), pc);
}

}

}
