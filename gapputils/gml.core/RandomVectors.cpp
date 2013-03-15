/*
 * RandomVectors.cpp
 *
 *  Created on: Feb 27, 2013
 *      Author: tombr
 */

#include "RandomVectors.h"

#include <tbblas/random.hpp>

namespace gml {
namespace core {

BeginPropertyDefinitions(RandomVectors)

  ReflectableBase(DefaultWorkflowElement<RandomVectors>)

  WorkflowProperty(Mean)
  WorkflowProperty(Stddev)
  WorkflowProperty(ElementCount)
  WorkflowProperty(VectorCount)
  WorkflowProperty(Vectors, Output("Vs"))

EndPropertyDefinitions

RandomVectors::RandomVectors() : _Mean(0), _Stddev(1), _ElementCount(0), _VectorCount(0) {
  setLabel("Random");
}

void RandomVectors::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  const double mean = getMean(), sd = getStddev();

  random_tensor<double, 1, false, normal<double> > randn(getElementCount());
  boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > vectors(
      new std::vector<boost::shared_ptr<std::vector<double> > >());
  for (int i = 0; i < getVectorCount(); ++i) {
    boost::shared_ptr<std::vector<double> > vector(new std::vector<double>(getElementCount()));
    thrust::copy((sd * randn + mean).begin(), (sd * randn + mean).end(), vector->begin());
    vectors->push_back(vector);
  }
  newState->setVectors(vectors);
}

} /* namespace core */
} /* namespace gml */
