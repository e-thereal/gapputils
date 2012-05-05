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

  DefineProperty(Filters, Serialize<TYPE_OF(Filters)>())
  DefineProperty(VisibleBias, Serialize<TYPE_OF(VisibleBias)>())
  DefineProperty(HiddenBiases, Serialize<TYPE_OF(HiddenBiases)>())
  DefineProperty(Mean, Serialize<TYPE_OF(Mean)>())
  DefineProperty(Stddev, Serialize<TYPE_OF(Stddev)>())
  DefineProperty(PoolingBlockSize, Serialize<TYPE_OF(PoolingBlockSize)>())
  DefineProperty(IsGaussian, Serialize<TYPE_OF(IsGaussian)>())

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

  return model;
}

} /* namespace ml */

} /* namespace gapputils */
