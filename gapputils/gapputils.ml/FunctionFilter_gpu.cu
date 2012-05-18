/*
 * FunctionFilter_gpu.cu
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT

#include "FunctionFilter.h"

#include <capputils/Verifier.h>
#include <culib/CudaImage.h>
#include <culib/math3d.h>

#include <thrust/transform.h>

namespace gapputils {

namespace ml {

struct gpu_log {
  __device__ float operator()(const float& x) const {
    if (x <= 0.f)
      return 0.f;
    else
      return logf(x);
  }
};

struct gpu_sqrt {
  __device__ float operator()(const float& x) const {
    if (x <= 0.f)
      return 0.f;
    else
      return sqrtf(x);
  }
};

struct gpu_bernstein {
  float coefficient, e1, e2;

  gpu_bernstein(float k, float n) {
    e1 = k;
    e2 = n - k;
    coefficient = binomial(n, k);
  }

  __device__ float operator()(const float& x) const {
    return coefficient * pow(x, e1) * pow(1.f - x, e2);
  }

};

void FunctionFilter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  if (!data)
    data = new FunctionFilter();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage())
    return;

  culib::ICudaImage& input = *getInputImage();
  boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(input.getSize(), input.getVoxelSize()));

  const unsigned count = input.getSize().x * input.getSize().y * input.getSize().z;

  thrust::device_ptr<float> inputPtr(input.getDevicePointer());
  thrust::device_ptr<float> outputPtr(output->getDevicePointer());

  switch(getFunction()) {
  case FilterFunction::Log:
    thrust::transform(inputPtr, inputPtr + count, outputPtr, gpu_log());
    break;

  case FilterFunction::Sqrt:
    thrust::transform(inputPtr, inputPtr + count, outputPtr, gpu_sqrt());
    break;

  case FilterFunction::Bernstein: {
    BernsteinParameters* params = dynamic_cast<BernsteinParameters*>(getParameters().get());
    if (params) {
      thrust::transform(inputPtr, inputPtr + count, outputPtr,
          gpu_bernstein(params->getIndex(), params->getDegree()));
    } else {

    }
    } break;
  }

  output->saveDeviceToOriginalImage();
  output->saveDeviceToWorkingCopy();

  input.freeCaches();
  output->freeCaches();

  data->setOutputImage(output);
}

}

}
