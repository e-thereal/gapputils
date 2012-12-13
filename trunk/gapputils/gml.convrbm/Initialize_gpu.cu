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
#include <tbblas/mask.hpp>
#include <tbblas/zeros.hpp>

namespace gml {

namespace convrbm {

InitializeChecker::InitializeChecker() {
  Initialize test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Tensors, test);
  CHECK_MEMORY_LAYOUT2(FilterWidth, test);
  CHECK_MEMORY_LAYOUT2(FilterHeight, test);
  CHECK_MEMORY_LAYOUT2(FilterCount, test);
  CHECK_MEMORY_LAYOUT2(Sigma, test);
  CHECK_MEMORY_LAYOUT2(WeightMean, test);
  CHECK_MEMORY_LAYOUT2(WeightStddev, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
}

void Initialize::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;

  // Calculate the mean and the std of all features
  const int filterCount = getFilterCount();

  boost::shared_ptr<Model> crbm(new Model());
  crbm->setVisibleUnitType(getVisibleUnitType());
  crbm->setHiddenUnitType(getHiddenUnitType());

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getTensors();

  if (getVisibleUnitType() == UnitType::Gaussian) {

    // Calculate the mean and normalize the data
    value_t mean = 0;
    for (size_t i = 0; i < tensors.size(); ++i)
      mean = mean + sum(*tensors[i]) / tensors[i]->count();
    mean /= tensors.size();

    // Calculate the stddev and normalize the data
    value_t var = 0;
    for (size_t i = 0; i < tensors.size(); ++i)
      var += dot(*tensors[i] - mean, *tensors[i] - mean) / tensors[i]->count();

    value_t stddev = sqrt(var / tensors.size());
    crbm->setMean(mean);
    crbm->setStddev(stddev);
  } else {
    crbm->setMean(0.0);
    crbm->setStddev(1.0);
  }

  // Initialize filters and bias terms
  boost::shared_ptr<tensor_t> vb(new tensor_t(zeros<value_t>(tensors[0]->size())));
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > hb(new std::vector<boost::shared_ptr<tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > filters(new std::vector<boost::shared_ptr<tensor_t> >());

  random_tensor<value_t, Model::dimCount, false, normal<value_t> > randn(tensors[0]->size());
  tensor<value_t, Model::dimCount, false> sample;

  tensor_t::dim_t maskSize = randn.size();
  maskSize[0] = getFilterWidth();
  maskSize[1] = getFilterHeight();

  tensor_t::dim_t hiddenSize = tensors[0]->size();
  hiddenSize[Model::dimCount - 1] = 1;

  for (int i = 0; i < filterCount; ++i) {
    if (getSigma() > 0.0) {
      sample = (getWeightStddev() * randn + getWeightMean()) * gaussian<value_t>(randn.size(), getSigma()) * mask<value_t>(randn.size(), maskSize);
    } else {
      sample = (getWeightStddev() * randn + getWeightMean()) * mask<value_t>(randn.size(), maskSize);
    }
    filters->push_back(boost::shared_ptr<tensor_t>(new tensor_t(sample)));
    hb->push_back(boost::make_shared<tensor_t>(zeros<value_t>(hiddenSize)));
  }

  crbm->setFilters(filters);
  crbm->setHiddenBiases(hb);
  crbm->setVisibleBias(vb);
  crbm->setFilterKernelSize(maskSize);

  newState->setModel(crbm);
}

}

}
