/*
 * Model.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#include "Model.h"

#include "tbblas_serialize.hpp"

using namespace capputils::attributes;

namespace gml {

namespace convrbm {

BeginPropertyDefinitions(Model)
  DefineProperty(Filters, Serialize<Type>())
  DefineProperty(VisibleBias, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
  DefineProperty(VisibleUnitType, Serialize<Type>())
  DefineProperty(HiddenUnitType, Serialize<Type>())
EndPropertyDefinitions

Model::Model() : _VisibleBias(0), _Mean(0.0), _Stddev(1.0) { }

Model::~Model() { }

boost::shared_ptr<Model> Model::clone() {
  boost::shared_ptr<Model> model(new Model());

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
  model->setVisibleUnitType(getVisibleUnitType());
  model->setHiddenUnitType(getHiddenUnitType());

  return model;
}

ModelChecker modelChecker;

}

} /* namespace gml */
