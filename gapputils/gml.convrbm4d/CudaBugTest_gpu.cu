/*
 * CudaBugTest_gpu.cu
 *
 *  Created on: Mar 8, 2013
 *      Author: tombr
 */

#include "CudaBugTest.h"

#include <cuda_runtime.h>

namespace gml {

namespace convrbm4d {

void CudaBugTest::update(IProgressMonitor* monitor) const {
  double* ptr;
  cudaMalloc(&ptr, 8);
  sleep(1);
  cudaFree(ptr);
//  cudaDeviceReset();
}

}

}
