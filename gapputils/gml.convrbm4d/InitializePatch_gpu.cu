/*
 * InitializePatch_gpu.cu
 *
 *  Created on: Oct 16, 2014
 *      Author: tombr
 */

#include "InitializePatch.h"

#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/random.hpp>
#include <tbblas/gaussian.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/rearrange.hpp>

namespace gml {

namespace convrbm4d {

InitializePatchChecker::InitializePatchChecker() {
  InitializePatch test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Tensors, test);
  CHECK_MEMORY_LAYOUT2(FilterWidth, test);
  CHECK_MEMORY_LAYOUT2(FilterHeight, test);
  CHECK_MEMORY_LAYOUT2(FilterDepth, test);
  CHECK_MEMORY_LAYOUT2(FilterCount, test);
  CHECK_MEMORY_LAYOUT2(StrideWidth, test);
  CHECK_MEMORY_LAYOUT2(StrideHeight, test);
  CHECK_MEMORY_LAYOUT2(StrideDepth, test);
  CHECK_MEMORY_LAYOUT2(WeightMean, test);
  CHECK_MEMORY_LAYOUT2(WeightStddev, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchChannels, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
  CHECK_MEMORY_LAYOUT2(ConvolutionType, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
}

void InitializePatch::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef host_tensor_t::value_t value_t;

  Logbook& dlog = getLogbook();

  // Calculate the mean and the std of all features
  const int filterCount = getFilterCount();
  const int dimCount = host_tensor_t::dimCount;

  boost::shared_ptr<model_t> crbm(new model_t());
  crbm->set_visibles_type(getVisibleUnitType());
  crbm->set_hiddens_type(getHiddenUnitType());
  crbm->set_convolution_type(getConvolutionType());

  host_tensor_t::dim_t stride = seq(getStrideWidth(), getStrideHeight(), getStrideDepth(), 1);

  v_host_tensor_t* tensors = getTensors().get();
  if (tensors && tensors->size() == 0)
    tensors = NULL;
  host_tensor_t::dim_t size = rearrange(ones<value_t>(getPatchWidth(), getPatchHeight(), getPatchDepth(), getPatchChannels()), stride).size(), maskSize = size;
  maskSize[dimCount - 1] = 1;

  host_tensor_t mask = ones<value_t>(maskSize);

  if (getVisibleUnitType() == unit_type::Gaussian && !tensors) {
    dlog(Severity::Warning) << "Training set required when visible units type is set to Gaussian. Aborting!";
    return;
  }

  const int totalCount = (getVisibleUnitType() == unit_type::Gaussian ? tensors->size() * 2 : 0) + filterCount;

  if (getVisibleUnitType() == unit_type::Gaussian) {

    // Calculate the mean and normalize the data
    value_t mean = 0;
    for (size_t i = 0; i < tensors->size(); ++i) {
      mean = mean + sum(*tensors->at(i)) / tensors->at(i)->count();
      if (monitor)
        monitor->reportProgress(100.0 * i / totalCount);
    }
    mean /= tensors->size();

    // Calculate the stddev and normalize the data
    value_t var = 0;
    for (size_t i = 0; i < tensors->size(); ++i) {
      var += dot(*tensors->at(i) - mean, *tensors->at(i) - mean) / tensors->at(i)->count();
      if (monitor)
        monitor->reportProgress(100.0 * (i + tensors->size()) / totalCount);
    }

    value_t stddev = sqrt(var / tensors->size());
    crbm->set_mean(mean);
    crbm->set_stddev(stddev);
  } else {
    crbm->set_mean(0.0);
    crbm->set_stddev(1.0);
  }

  // Initialize filters and bias terms
  host_tensor_t vb = zeros<value_t>(size);
  v_host_tensor_t hb;
  v_host_tensor_t filters;

  if ((getFilterWidth() % stride[0]) != 0 ||
      (getFilterHeight() % stride[1]) != 0 ||
      (getFilterDepth() % stride[2]) != 0)
  {
    dlog(Severity::Warning) << "Filter size must be a multiple of stride size. Aborting!";
    return;
  }

  host_tensor_t::dim_t kernelSize;
  kernelSize[0] = getFilterWidth() / stride[0];
  kernelSize[1] = getFilterHeight() / stride[1];
  kernelSize[2] = getFilterDepth() / stride[2];
  kernelSize[3] = size[3];

  random_tensor<value_t, model_t::dimCount, false, normal<value_t> > randn(kernelSize);
  host_tensor_t sample;

  for (int i = 0; i < filterCount; ++i) {
    sample = (getWeightStddev() * randn + getWeightMean()); // / (value_t)randn.count();
    filters.push_back(boost::make_shared<host_tensor_t>(sample));
    hb.push_back(boost::make_shared<host_tensor_t>(zeros<value_t>(maskSize)));
    if (monitor)
      monitor->reportProgress(100.0 * (i + (getVisibleUnitType() == unit_type::Gaussian ? 2 * tensors->size() : 0)) / totalCount);
  }

  crbm->set_filters(filters);
  crbm->set_hidden_bias(hb);
  crbm->set_visible_bias(vb);
  crbm->set_kernel_size(kernelSize);
  crbm->set_stride_size(stride);
  crbm->set_mask(mask);

  newState->setModel(crbm);
}

}

}
