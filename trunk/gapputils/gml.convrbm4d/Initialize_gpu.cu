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

  typedef host_tensor_t::value_t value_t;

  Logbook& dlog = getLogbook();

  // Calculate the mean and the std of all features
  const int filterCount = getFilterCount();
  const int dimCount = host_tensor_t::dimCount;

  boost::shared_ptr<model_t> crbm(new model_t());
  crbm->set_visibles_type(getVisibleUnitType());
  crbm->set_hiddens_type(getHiddenUnitType());
  crbm->set_convolution_type(getConvolutionType());

  v_host_tensor_t& tensors = *getTensors();
  host_tensor_t::dim_t size = tensors[0]->size(), maskSize = size;
  maskSize[dimCount - 1] = 1;

  host_tensor_t mask = (getMask() ? *getMask() : ones<value_t>(maskSize));
  const value_t count = sum(mask) * size[dimCount - 1];

  if (!(mask.size() == maskSize)) {
    dlog(Severity::Warning) << "Size mismatch between input tensors and mask. Aborting!";
    return;
  }

  const int totalCount = tensors.size() * 2 + filterCount;

  if (getVisibleUnitType() == unit_type::Gaussian) {

    // Calculate the mean and normalize the data
    value_t mean = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      mean = mean + sum(*tensors[i] * repeat(mask, size / maskSize)) / count;
      if (monitor)
        monitor->reportProgress(100.0 * i / totalCount);
    }
    mean /= tensors.size();

    // Calculate the stddev and normalize the data
    value_t var = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      var += dot((*tensors[i] - mean) * repeat(mask, size / maskSize), (*tensors[i] - mean) * repeat(mask, size / maskSize)) / count;
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
  host_tensor_t vb = zeros<value_t>(tensors[0]->size());
  v_host_tensor_t hb;
  v_host_tensor_t filters;

  host_tensor_t::dim_t kernelSize;
  kernelSize[0] = getFilterWidth();
  kernelSize[1] = getFilterHeight();
  kernelSize[2] = getFilterDepth();
  kernelSize[3] = size[3];

  random_tensor<value_t, model_t::dimCount, false, normal<value_t> > randn(kernelSize);
  host_tensor_t sample;

  host_tensor_t::dim_t hiddenSize = tensors[0]->size();
  hiddenSize[model_t::dimCount - 1] = 1;

  for (int i = 0; i < filterCount; ++i) {
    sample = (getWeightStddev() * randn + getWeightMean()); // / (value_t)randn.count();
    filters.push_back(boost::make_shared<host_tensor_t>(sample));
    hb.push_back(boost::make_shared<host_tensor_t>(zeros<value_t>(hiddenSize)));
    if (monitor)
      monitor->reportProgress(100.0 * (i + 2 * tensors.size()) / totalCount);
  }

  crbm->set_filters(filters);
  crbm->set_hidden_bias(hb);
  crbm->set_visible_bias(vb);
  crbm->set_kernel_size(kernelSize);
  crbm->set_mask(mask);

  newState->setModel(crbm);
}

}

}
