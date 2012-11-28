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

namespace gml {

namespace convrbm {

InitializeChecker::InitializeChecker() {
  Initialize test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Tensors, test);
  CHECK_MEMORY_LAYOUT2(FilterWidth, test);
  CHECK_MEMORY_LAYOUT2(FilterHeight, test);
  CHECK_MEMORY_LAYOUT2(FilterCount, test);
  CHECK_MEMORY_LAYOUT2(WeightMean, test);
  CHECK_MEMORY_LAYOUT2(WeightStddev, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
}

void Initialize::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;

  // Calculate the mean and the std of all features
  const unsigned filterCount = getFilterCount();
  const int filterWidth = getFilterWidth();
  const int filterHeight = getFilterHeight();
  const int filterDepth = getTensors()->at(0)->size()[2];

  boost::shared_ptr<Model> crbm(new Model());
  crbm->setVisibleUnitType(getVisibleUnitType());
  crbm->setHiddenUnitType(getHiddenUnitType());

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getTensors();

  if (getVisibleUnitType() == UnitType::Gaussian) {

    // Calculate the mean and normalize the data
    value_t mean = 0;
    for (unsigned i = 0; i < tensors.size(); ++i)
      mean = mean + sum(*tensors[i]) / tensors[i]->count();
    mean /= tensors.size();

    // Calculate the stddev and normalize the data
    value_t var = 0;
    for (unsigned i = 0; i < tensors.size(); ++i)
      var += dot(*tensors[i] - mean, *tensors[i] - mean) / tensors[i]->count();

    value_t stddev = sqrt(var / tensors.size());
    crbm->setMean(mean);
    crbm->setStddev(stddev);
  } else {
    crbm->setMean(0.0);
    crbm->setStddev(1.0);
  }

  // Initialize filters and bias terms
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > filters(new std::vector<boost::shared_ptr<tensor_t> >());
  random_tensor<value_t, Model::dimCount, false, normal<value_t> > randn(filterWidth, filterHeight, filterDepth);
  tensor<value_t, Model::dimCount, false> sample;

  for (unsigned i = 0; i < filterCount; ++i) {
    sample = getWeightStddev() * randn + getWeightMean();
    filters->push_back(boost::shared_ptr<tensor_t>(new tensor_t(sample)));
  }

  crbm->setFilters(filters);
  crbm->setHiddenBiases(boost::shared_ptr<std::vector<value_t> >(new std::vector<value_t>(filterCount, 0)));
  crbm->setVisibleBias(0);

  newState->setModel(crbm);
}

}

}
