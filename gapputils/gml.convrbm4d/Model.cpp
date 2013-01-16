/*
 * Model.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#include "Model.h"

#include "tbblas_serialize.hpp"

#include <boost/make_shared.hpp>

using namespace capputils::attributes;

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Model)
  DefineProperty(Filters, Serialize<Type>())
  DefineProperty(VisibleBias, Serialize<Type>())
  DefineProperty(HiddenBiases, Serialize<Type>())
  DefineProperty(FilterKernelSize, Serialize<Type>())
  DefineProperty(Mean, Serialize<Type>())
  DefineProperty(Stddev, Serialize<Type>())
  DefineProperty(VisibleUnitType, Serialize<Type>())
  DefineProperty(HiddenUnitType, Serialize<Type>())
EndPropertyDefinitions

Model::Model() : _Mean(0.0), _Stddev(1.0) { }

boost::shared_ptr<Model> Model::clone() {
  boost::shared_ptr<Model> model(new Model());

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > filters(new std::vector<boost::shared_ptr<tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > hb(new std::vector<boost::shared_ptr<tensor_t> >());
  std::vector<boost::shared_ptr<tensor_t> >& oldfilters = *getFilters();
  std::vector<boost::shared_ptr<tensor_t> >& oldhb = *getHiddenBiases();

  assert(oldfilters.size() == oldhb.size());
  for (unsigned i = 0; i < oldfilters.size(); ++i) {
    tensor_t& oldfilter = *oldfilters[i];
    boost::shared_ptr<tensor_t> filter(new tensor_t(oldfilter.size()));
    std::copy(oldfilter.begin(), oldfilter.end(), filter->begin());
    filters->push_back(filter);

    hb->push_back(boost::make_shared<tensor_t>(*oldhb[i]));
  }

  model->setFilters(filters);
  model->setVisibleBias(boost::make_shared<tensor_t>(*getVisibleBias()));
  model->setHiddenBiases(hb);
  model->setFilterKernelSize(getFilterKernelSize());
  model->setMean(getMean());
  model->setStddev(getStddev());
  model->setVisibleUnitType(getVisibleUnitType());
  model->setHiddenUnitType(getHiddenUnitType());

  return model;
}

ModelChecker modelChecker;

}

} /* namespace gml */
