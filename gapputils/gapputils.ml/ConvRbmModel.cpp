/*
 * ConvRbmModel.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#include "ConvRbmModel.h"

#include <algorithm>

#include "tbblas_serialize.hpp"

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(ConvRbmModel)
  using namespace capputils::attributes;

  DefineProperty(Filters, Serialize<Type>())
  DefineProperty(VisibleBias, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
  DefineProperty(PoolingBlockSize, Serialize<Type>())
  DefineProperty(IsGaussian, Serialize<Type>())
  DefineProperty(HiddenUnitType, Serialize<Type>())

EndPropertyDefinitions

ConvRbmModel::ConvRbmModel() {
}

ConvRbmModel::~ConvRbmModel() {
}

boost::shared_ptr<ConvRbmModel> ConvRbmModel::clone() {
  boost::shared_ptr<ConvRbmModel> model(new ConvRbmModel());

//  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
//  Property(VisibleBias, value_t)
//  Property(HiddenBiases, boost::shared_ptr<std::vector<value_t> >)
//  Property(Mean, value_t)
//  Property(Stddev, value_t)
//  Property(IsGaussian, bool)

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > filters(new std::vector<boost::shared_ptr<tensor_t> >());
  std::vector<boost::shared_ptr<tensor_t> >& oldfilters = *getFilters();
  for (unsigned i = 0; i < oldfilters.size(); ++i) {
    tensor_t& oldfilter = *oldfilters[i];
    boost::shared_ptr<tensor_t> filter(new tensor_t(oldfilter.size()));
    std::copy(oldfilter.begin(), oldfilter.end(), filter->begin());
    filters->push_back(filter);
  }
  model->setFilters(filters);
  model->setVisibleBias(getVisibleBias());
  model->setHiddenBiases(boost::shared_ptr<std::vector<value_t> >(new std::vector<value_t>(
      getHiddenBiases()->begin(), getHiddenBiases()->end())));
  model->setMean(getMean());
  model->setStddev(getStddev());
  model->setPoolingBlockSize(getPoolingBlockSize());
  model->setIsGaussian(getIsGaussian());
  model->setHiddenUnitType(getHiddenUnitType());

  return model;
}

} /* namespace ml */

} /* namespace gapputils */
