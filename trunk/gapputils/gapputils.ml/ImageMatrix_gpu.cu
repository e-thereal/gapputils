#define BOOST_TYPEOF_COMPLIANT

#include "ImageMatrix.h"

#include <capputils/Verifier.h>
#include <culib/CudaImage.h>

#include "../tbblas/device_matrix.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include <cassert>
#include <iostream>

//#define TRACE std::cout << __LINE__ << std::endl;

namespace gapputils {

namespace ml {

template<class T>
struct axpy : public thrust::unary_function<T, T> {
public:
  typedef T result_type;

private:
  T a, y;

public:
  axpy(const T& a, const T& y) : a(a), y(y) { }

  __device__ __host__
  T operator()(const T& x) const {
    return a * x + y;
  }
};

void ImageMatrix::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageMatrix();

  if (!capputils::Verifier::Valid(*this) || !getInputImage())
    return;

  dim3 size = getInputImage()->getSize();
  int rowCount = ceil(std::sqrt((float)size.z));

  boost::shared_ptr<culib::ICudaImage> imageMatrix(new culib::CudaImage(dim3(size.x * rowCount, size.y * rowCount)));
  
  // anneliebttom copy this

  culib::ICudaImage& input = *getInputImage();
  dim3 inSize = input.getSize();
  dim3 outSize = imageMatrix->getSize();
  const int count = inSize.x * inSize.y;
  thrust::device_ptr<float> inputPtr(input.getDevicePointer());

  assert(inSize.x * rowCount == outSize.x);
  assert(inSize.y * rowCount == outSize.y);
  assert(outSize.z == 1);

  // Copy each image into the device matrix
  // Copy device matrix to cuda image

  std::cout << "Size: " << outSize.x << ", " << outSize.y << std::endl;

  tbblas::device_matrix<float> m(outSize.x, outSize.y);
  for (int y = 0, i = 0; y < rowCount; ++y) {
    for (int x = 0; x < rowCount && i < inSize.z; ++x, ++i) {
      tbblas::device_matrix<float> submatrix = tbblas::subrange(m, x * inSize.x, (x + 1) * inSize.x,
          y * inSize.y, (y + 1) * inSize.y);

      float a = 1.f / (getMaxValue() - getMinValue());
      float y = -getMinValue() * a;
  
      thrust::copy(thrust::make_transform_iterator(inputPtr + i * count, axpy<float>(a, y)),
          thrust::make_transform_iterator(inputPtr + (i + 1) * count, axpy<float>(a, y)),
          submatrix.begin());
    }
  }
  
  thrust::copy(m.data().begin(), m.data().end(), imageMatrix->getWorkingCopy());
  data->setImageMatrix(imageMatrix);
}

}

}