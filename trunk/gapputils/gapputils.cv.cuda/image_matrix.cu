#include "image_matrix.hpp"

#define BOOST_TYPEOF_COMPLIANT

#include "../tbblas/device_matrix.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <cassert>
#include <iostream>

namespace gapputils {

namespace cv {

namespace cuda {

#define TRACE std::cout << __LINE__ << std::endl;

void createImageMatrix(culib::ICudaImage& input, culib::ICudaImage& imageMatrix) {
  dim3 inSize = input.getSize();TRACE
  dim3 outSize = imageMatrix.getSize();TRACE
  int rowCount = ceil(std::sqrt((float)inSize.z));TRACE
  const int count = inSize.x * inSize.y;TRACE
  thrust::device_ptr<float> inputPtr(input.getDevicePointer());TRACE

  assert(inSize.x * rowCount == outSize.x);TRACE
  assert(inSize.y * rowCount == outSize.y);TRACE
  assert(outSize.z == 1);TRACE

  // Copy each image into the device matrix
  // Copy device matrix to cuda image

  std::cout << "Size: " << outSize.x << ", " << outSize.y << std::endl;

  tbblas::device_matrix<float> matrix(outSize.x, outSize.y);TRACE
  for (int y = 0, i = 0; y < rowCount; ++y) {TRACE
    for (int x = 0; x < rowCount && i < inSize.z; ++x, ++i) {TRACE
      tbblas::device_matrix<float> submatrix = tbblas::subrange(matrix, x * inSize.x, (x + 1) * inSize.x,
        y * inSize.y, (y + 1) * inSize.y);
  TRACE
      thrust::copy(inputPtr + i * count, inputPtr + (i + 1) * count, submatrix.begin());TRACE
    }
  }
  TRACE
  thrust::copy(matrix.data().begin(), matrix.data().end(), imageMatrix.getDevicePointer());TRACE
  imageMatrix.saveDeviceToWorkingCopy();TRACE
  imageMatrix.freeCaches();TRACE
}

}

}

}