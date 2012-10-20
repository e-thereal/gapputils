/*
 * FunctionFilter_gpu.cu
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT

#include "FunctionFilter.h"

#include <capputils/Verifier.h>
#include <culib/math3d.h>

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

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

struct gpu_gamma {
  float slope, gamma, intercept;

  gpu_gamma(const float& slope, const float& gamma, const float& intercept) : slope(slope), gamma(gamma), intercept(intercept) { }

  __device__ float operator()(const float& x) const {
    return slope * powf(x, gamma) + intercept;
  }
};

struct gpu_sigmoid {
  float slope, inflection, minimum, invContrast;

  gpu_sigmoid(const float& slope, const float& inflection) : slope(slope), inflection(inflection) {
    minimum = 1.f / (1.f + expf(slope * inflection));
    const float maximum = 1.f / (1.f + expf(-slope * (1.f - inflection)));
    invContrast = 1.f / (maximum - minimum);
  }

  __device__ float operator()(const float& x) const {
    return (1.f / (1.f + expf(-slope * (x - inflection))) - minimum) * invContrast;
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

  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  const unsigned count = input.getSize()[0] * input.getSize()[1] * input.getSize()[2];

  thrust::device_vector<float> data(input.getData(), input.getData() + count);

  switch(getFunction()) {
  case FilterFunction::Log:
    thrust::transform(data.begin(), data.end(), data.begin(), gpu_log());
    break;

  case FilterFunction::Sqrt:
    thrust::transform(data.begin(), data.end(), data.begin(), gpu_sqrt());
    break;

  case FilterFunction::Bernstein: {
    BernsteinParameters* params = dynamic_cast<BernsteinParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(),
          gpu_bernstein(params->getIndex(), params->getDegree()));
    }
    } break;

  case FilterFunction::Gamma: {
    GammaParameters* params = dynamic_cast<GammaParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(),
        gpu_gamma(params->getSlope(), params->getGamma(), params->getIntercept()));
    }
    } break;

  case FilterFunction::Sigmoid: {
    SigmoidParameters* params = dynamic_cast<SigmoidParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(),
        gpu_sigmoid(params->getSlope(), params->getInflection()));
    }
    } break;

  case FilterFunction::Threshold: {
    ThresholdParameters* params = dynamic_cast<ThresholdParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(), _1 > params->getThreshold());
    }
    } break;
  }

  thrust::copy(data.begin(), data.end(), output->getData());
  this->data->setOutputImage(output);
}

}

}
