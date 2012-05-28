#include "aggregate.h"

#include <thrust/transform.h>

#include <cassert>

namespace gapputils {

namespace cv {

namespace cuda {

struct sum_kernel {
  int pitch, depth;

  sum_kernel(int pitch, int depth) : pitch(pitch), depth(depth) { }

  __device__ float operator()(const float& x) const {
    float result = 0;
    for (int i = 0; i < depth; ++i) {
      result += *(&x + i * pitch);
    }
    return result;
  }
};

struct average_kernel {
  int pitch, depth;

  average_kernel(int pitch, int depth) : pitch(pitch), depth(depth) { }

  __device__ float operator()(const float& x) const {
    float result = 0;
    for (int i = 0; i < depth; ++i) {
      result += *(&x + i * pitch);
    }
    return result / depth;
  }
};

void sum(culib::ICudaImage* input, culib::ICudaImage* output) {
  thrust::device_ptr<float> in(input->getDevicePointer());
  thrust::device_ptr<float> out(output->getDevicePointer());

  assert(input->getSize().x == output->getSize().x);
  assert(input->getSize().y == output->getSize().y);
  assert(output->getSize().z == 1);

  const int pitch = input->getSize().x * input->getSize().y;
  const int depth = input->getSize().z;

  thrust::transform(in, in + pitch, out, sum_kernel(pitch, depth));
}

void average(culib::ICudaImage* input, culib::ICudaImage* output) {
  thrust::device_ptr<float> in(input->getDevicePointer());
  thrust::device_ptr<float> out(output->getDevicePointer());

  assert(input->getSize().x == output->getSize().x);
  assert(input->getSize().y == output->getSize().y);
  assert(output->getSize().z == 1);

  const int pitch = input->getSize().x * input->getSize().y;
  const int depth = input->getSize().z;

  thrust::transform(in, in + pitch, out, average_kernel(pitch, depth));
}

}

}

}
