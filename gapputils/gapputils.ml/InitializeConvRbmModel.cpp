/*
 * InitializeConvRbmModel.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#include "InitializeConvRbmModel.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <iostream>
#include <curand.h>

#include "sampling.hpp"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(InitializeConvRbmModel)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputTensors, Input("Imgs"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(FilterCount, Observe(PROPERTY_ID))
  DefineProperty(FilterSize, Observe(PROPERTY_ID))
  DefineProperty(PoolingBlockSize, Observe(PROPERTY_ID))
  DefineProperty(WeightMean, Observe(PROPERTY_ID))
  DefineProperty(WeightStddev, Observe(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID))

  DefineProperty(Model, Output("CRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl

InitializeConvRbmModel::InitializeConvRbmModel() : data(0) {
  WfeUpdateTimestamp
  setLabel("InitializeConvRbmModel");

//  ConvRbmModel test;
//  LOCATE(test, Filters);
//  LOCATE(test, VisibleBias);
//  LOCATE(test, HiddenBiases);
//  LOCATE(test, Mean);
//  LOCATE(test, Stddev);
//  LOCATE(test, IsGaussian);

  Changed.connect(capputils::EventHandler<InitializeConvRbmModel>(this, &InitializeConvRbmModel::changedHandler));
}

InitializeConvRbmModel::~InitializeConvRbmModel() {
  if (data)
    delete data;
}

void InitializeConvRbmModel::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void InitializeConvRbmModel::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  if (!data)
    data = new InitializeConvRbmModel();

//  ConvRbmModel test;
//  LOCATE(test, Filters);
//  LOCATE(test, VisibleBias);
//  LOCATE(test, HiddenBiases);
//  LOCATE(test, Mean);
//  LOCATE(test, Stddev);
//  LOCATE(test, IsGaussian);

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputTensors() || getInputTensors()->size() == 0) {
    std::cout << "[Warning] Missing training set!" << std::endl;
    return;
  }

  std::cout << "Initializing ConvRBM ..." << std::endl;

  // Calculate the mean and the std of all features
  const unsigned filterCount = getFilterCount();
  const unsigned filterSize = getFilterSize();
  const unsigned filterDepth = getInputTensors()->at(0)->size()[2];
  const unsigned sampleCount = getInputTensors()->size();

  boost::shared_ptr<ConvRbmModel> crbm(new ConvRbmModel());
  crbm->setPoolingBlockSize(getPoolingBlockSize());
  crbm->setIsGaussian(getIsGaussian());

  if (getIsGaussian()) {
    // calculate mean and stddev and normalize data
    assert(0); //< Gaussian ConvRBM not yet implemented.
  } else {
    crbm->setMean(0.0);
    crbm->setStddev(1.0);
  }

  curandGenerator_t gen;
  curandStatus_t status;
  if (curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
    std::cout << "[Warning] Could not create random number generator." << std::endl;

  // Initialize filters and bias terms
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > filters(new std::vector<boost::shared_ptr<tensor_t> >());

  const int filterVoxelCount = filterSize * filterSize * filterDepth;
  std::vector<value_t> values(filterVoxelCount + (filterVoxelCount % 2));
  for (unsigned i = 0; i < filterCount; ++i) {
    boost::shared_ptr<tensor_t> filter(new tensor_t(filterSize, filterSize, filterDepth));

    if ((status = curandGenerateNormalDouble(gen,
        &values[0],
        values.size(),
        getWeightMean(), getWeightStddev())) != CURAND_STATUS_SUCCESS)
    {
      std::cout << "[Warning] Could not generate random numbers: " << status << std::endl;
    }
    std::copy(values.begin(), values.begin() + filterVoxelCount, filter->data().begin());

//    thrust::transform(thrust::counting_iterator<unsigned>(i), thrust::counting_iterator<unsigned>(filter->data().size() + i),
//        filter->begin(), get_randn<double>(0.f, getWeightStddev()));
    filters->push_back(filter);
//    std::cout << "FilterSum: " << tbblas::sum(*filter) << ", " << tbblas::dot(*filter, *filter) << std::endl;
//    std::cout << "Mean:   " << getWeightMean() << std::endl;
//    std::cout << "Stddev: " << getWeightStddev() << std::endl;
  }

  curandDestroyGenerator(gen);

  crbm->setFilters(filters);

  crbm->setHiddenBiases(boost::shared_ptr<std::vector<value_t> >(new std::vector<value_t>(filterCount, 0)));
  crbm->setVisibleBias(0);

  std::cout << "[Info] ConvRBM initialized." << std::endl;

  data->setModel(crbm);
}

void InitializeConvRbmModel::writeResults() {
  if (!data)
    return;

  setModel(data->getModel());
}

}

}
