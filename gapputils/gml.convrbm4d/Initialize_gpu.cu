/*
 * Initialize_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Initialize.h"

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

InitializeChecker::InitializeChecker() {
  Initialize test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Tensors, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
  CHECK_MEMORY_LAYOUT2(FilterWidth, test);
  CHECK_MEMORY_LAYOUT2(FilterHeight, test);
  CHECK_MEMORY_LAYOUT2(FilterDepth, test);
  CHECK_MEMORY_LAYOUT2(FilterCount, test);
  CHECK_MEMORY_LAYOUT2(StrideWidth, test);
  CHECK_MEMORY_LAYOUT2(StrideHeight, test);
  CHECK_MEMORY_LAYOUT2(StrideDepth, test);
  CHECK_MEMORY_LAYOUT2(PoolingMethod, test);
  CHECK_MEMORY_LAYOUT2(PoolingWidth, test);
  CHECK_MEMORY_LAYOUT2(PoolingHeight, test);
  CHECK_MEMORY_LAYOUT2(PoolingDepth, test);
  CHECK_MEMORY_LAYOUT2(WeightMean, test);
  CHECK_MEMORY_LAYOUT2(WeightStddev, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
  CHECK_MEMORY_LAYOUT2(ConvolutionType, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
}

void Initialize::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  const int dimCount = host_tensor_t::dimCount;

  typedef host_tensor_t::value_t value_t;
  typedef tensor<value_t, dimCount> tensor_t;

  Logbook& dlog = getLogbook();

  // Calculate the mean and the std of all features
  const int filterCount = getFilterCount();

  boost::shared_ptr<model_t> crbm(new model_t());
  crbm->set_visibles_type(getVisibleUnitType());
  crbm->set_hiddens_type(getHiddenUnitType());
  crbm->set_convolution_type(getConvolutionType());

  host_tensor_t::dim_t stride = seq(getStrideWidth(), getStrideHeight(), getStrideDepth(), 1);
  host_tensor_t::dim_t pooling = seq(getPoolingWidth(), getPoolingHeight(), getPoolingDepth(), 1);

  crbm->set_pooling_method(_PoolingMethod);
  crbm->set_pooling_size(pooling);

  v_host_tensor_t& tensors = *getTensors();
  host_tensor_t::dim_t size = tensors[0]->size(), maskSize = size;
  maskSize[dimCount - 1] = 1;

  host_tensor_t mask;
  if (getMask()) {
    mask = *getMask();
  } else {
    mask = ones<value_t>(maskSize);
  }

  const value_t count = sum(mask) * size[dimCount - 1];

  if (!(mask.size() == maskSize)) {
    dlog(Severity::Warning) << "Size mismatch between input tensors and mask. Aborting!";
    return;
  }

  if ((getFilterWidth()  % stride[0]) != (size[0] % stride[0]) ||
      (getFilterHeight() % stride[1]) != (size[1] % stride[1]) ||
      (getFilterDepth()  % stride[2]) != (size[2] % stride[2]))
  {
    dlog(Severity::Warning) << "Filter sizes must be congruent to the image sizes modulo the stride size. Aborting!";
    return;
  }

  const int totalCount = tensors.size() * 2 + filterCount;

  tensor_t tensor;

  if (getVisibleUnitType() == unit_type::Gaussian) {

    // Calculate the mean and normalize the data
    value_t mean = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensor = *tensors[i];
      mean = mean + sum(tensor * repeat(mask, size / maskSize)) / count;
      if (monitor)
        monitor->reportProgress(100.0 * i / totalCount);
    }
    mean /= tensors.size();

    // Calculate the stddev and normalize the data
    value_t var = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensor = *tensors[i];
      var += dot((tensor - mean) * repeat(mask, size / maskSize), (tensor - mean) * repeat(mask, size / maskSize)) / count;
      if (monitor)
        monitor->reportProgress(100.0 * (i + tensors.size()) / totalCount);
    }

    value_t stddev = sqrt(var / tensors.size());
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

  host_tensor_t::dim_t kernelSize = seq(getFilterWidth(), getFilterHeight(), getFilterDepth(), size[dimCount - 1]);

  random_tensor<value_t, model_t::dimCount, false, normal<value_t> > randn(kernelSize);
  host_tensor_t sample;

  host_tensor_t::dim_t hidden_mask_size = (size + stride - 1) / stride;
  hidden_mask_size[dimCount - 1] = 1;

  for (int i = 0; i < filterCount; ++i) {
    sample = (getWeightStddev() * randn + getWeightMean()); // / (value_t)randn.count();
    filters.push_back(boost::make_shared<host_tensor_t>(sample));
    hb.push_back(boost::make_shared<host_tensor_t>(zeros<value_t>(hidden_mask_size)));
    if (monitor)
      monitor->reportProgress(100.0 * (i + 2 * tensors.size()) / totalCount);
  }

  crbm->set_filters(filters);
  crbm->set_hidden_bias(hb);
  crbm->set_visible_bias(vb);
  crbm->set_kernel_size(kernelSize);
  crbm->set_stride_size(stride);
  crbm->set_mask(mask);

  if (!crbm->is_valid()) {
    dlog(Severity::Warning) << "Invalid parameters. Aborting!";
    return;
  }

  newState->setModel(crbm);
}

}

}
