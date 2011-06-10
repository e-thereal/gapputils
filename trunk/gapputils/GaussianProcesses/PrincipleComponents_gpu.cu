/*
 * PrincipleComponents_gpu.cu
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#include <culapackdevice.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <iostream>

namespace GaussianProcesses {

void getPcs(double* pc, double* data, int m, int n) {
  const int count = m * n;
  culaStatus status;

  float* f_data = new float[count];
  thrust::copy(data, data + count, f_data);

  thrust::device_vector<float> d_data(f_data, f_data + count);
  thrust::device_vector<float> d_U(m * m);
  thrust::device_vector<float> d_sigma(n);
  thrust::device_vector<float> d_Vt(n * n);

  if ((status = culaInitialize()) != culaNoError) {
    std::cout << "Could not initialize: " << culaGetStatusString(status) << std::endl;
    delete f_data;
    return;
  }

  //culaSgesvd
  char jobu = 'A';
  char jobvt = 'A';
  if ((status = culaDeviceSgesvd(jobu, jobvt, m, n, d_data.data().get(), m, d_sigma.data().get(),
      d_U.data().get(), m, d_Vt.data().get(), n)) != culaNoError)
  {
    std::cout << "Could not SVD: " << culaGetStatusString(status) << std::endl;
    delete f_data;
    return;
  }

  thrust::copy(d_U.begin(), d_U.end(), pc);
  //thrust::copy(data, data + count, pc);

  culaShutdown();

  delete f_data;
}

}
