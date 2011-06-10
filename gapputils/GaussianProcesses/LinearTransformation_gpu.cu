/*
 * LinearTransformation_gpu.cu
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#include <thrust/device_vector.h>
#include <cublas.h>

namespace GaussianProcesses {

void lintrans(double* output, int outM, int n, double* input,
    int inM, double* transformation, bool transpose)
{
  float* f_input = new float[n * inM];
  float* f_trans = new float[inM * outM];
  thrust::copy(input, input + (n * inM), f_input);
  thrust::copy(transformation, transformation + (inM * outM), f_trans);

  thrust::device_vector<float> d_input(f_input, f_input + (n * inM));
  thrust::device_vector<float> d_trans(f_trans, f_trans + (inM * outM));
  thrust::device_vector<float> d_output(n * outM);

  //cublasSgemm()
  if (transpose == false) {
    cublasSgemm('n', 'n', outM, n, inM, 1.0f, d_trans.data().get(), outM,
        d_input.data().get(), inM, 0.0f, d_output.data().get(), outM);
  } else {
    cublasSgemm('t', 'n', outM, n, inM, 1.0f, d_trans.data().get(), inM,
        d_input.data().get(), inM, 0.0f, d_output.data().get(), outM);
  }

  thrust::copy(d_output.begin(), d_output.end(), output);

  delete f_input;
  delete f_trans;
}

}
