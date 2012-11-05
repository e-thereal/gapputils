#define BOOST_TYPEOF_COMPLIANT

#include "ImageMatrix.h"

#include <capputils/Verifier.h>
#include <culib/CudaImage.h>
#include <culib/transform.h>
#include <culib/CulibException.h>

#include "../tbblas/device_matrix.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <cassert>
#include <iostream>

#include "cuda_util.h"

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
    return min((T)1, max((T)0, a * x + y));
  }
};

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl
#define LOCATE2(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

void ImageMatrix::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageMatrix();

  if (!capputils::Verifier::Valid(*this) || !getInputImage())
    return;

  boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImage());
  dim3 size = input->getSize();
  int columnCount = getColumnCount() > 0 ? getColumnCount() : ceil(std::sqrt((float)size.z));
  int rowCount = ceil((float)size.z / (float)columnCount);

//  std::cout << "ColumnCount: " << columnCount << " (" << getColumnCount() << ")" << std::endl;
//  std::cout << "RowCount: " << rowCount << std::endl;
//  std::cout << "Count: " << size.z << std::endl;

  boost::shared_ptr<image_t> outMatrix(new image_t(size.x * columnCount, size.y * rowCount, 1));
  boost::shared_ptr<culib::ICudaImage> imageMatrix(new culib::CudaImage(dim3(size.x * columnCount, size.y * rowCount)));
  
  // anneliebttom copy this

  dim3 inSize = input->getSize();
  dim3 outSize = imageMatrix->getSize();
  const int count = inSize.x * inSize.y;
  const int totalCount = inSize.x * inSize.y * inSize.z;


  // Center filter
  culib::CudaImage centered(inSize);
//  fmatrix4 centering = make_fmatrix4_translation(-size.x / 2, -size.y / 2, 0);
  fmatrix4 centering = make_fmatrix4_translation(-(int)size.x/2, -(int)size.y/2, 0);
  culib::transform3D(centered.getDevicePointer(), input->getCudaArray(), inSize, centering, dim3(), true);
  thrust::device_ptr<float> inputPtr(getCenterImages() ? centered.getDevicePointer() : input->getDevicePointer());

  assert(inSize.x * columnCount == outSize.x);
  assert(inSize.y * rowCount == outSize.y);
  assert(outSize.z == 1);

  // Copy each image into the device matrix
  // Copy device matrix to cuda image

//  std::cout << "Size: " << outSize.x << ", " << outSize.y << std::endl;

  float minV = getMinValue();
  float maxV = getMaxValue();

  if (getAutoScale()) {
    float _min = thrust::reduce(inputPtr, inputPtr + totalCount, input->getWorkingCopy()[0], thrust::minimum<float>());
    float _max = thrust::reduce(inputPtr, inputPtr + totalCount, input->getWorkingCopy()[0], thrust::maximum<float>());
//    maxV = min(abs(_max), abs(_min));
//    minV = -maxV;
    minV = _min;
    maxV = _max;
//    std::cout << input.getWorkingCopy()[0] << " in [" << _min << ", " << _max << "]" << std::endl;
  }

  float a = 1.f / (maxV - minV);
  float _y = -minV * a;

  tbblas::device_matrix<float> m(outSize.x, outSize.y);
  for (int y = 0, i = 0; y < rowCount; ++y) {
    for (int x = 0; x < columnCount && i < inSize.z; ++x, ++i) {
      tbblas::device_matrix<float> submatrix = tbblas::subrange(m, x * inSize.x, (x + 1) * inSize.x,
          y * inSize.y, (y + 1) * inSize.y);
  
      thrust::copy(thrust::make_transform_iterator(inputPtr + i * count, axpy<float>(a, _y)),
          thrust::make_transform_iterator(inputPtr + (i + 1) * count, axpy<float>(a, _y)),
          submatrix.begin());
    }
  }
  input->freeCaches();
  thrust::copy(m.data().begin(), m.data().end(), outMatrix->getData());
  data->setImageMatrix(outMatrix);
}

}

}
