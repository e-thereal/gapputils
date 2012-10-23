/*
 * BernoulliSampler.cpp
 *
 *  Created on: Oct 22, 2012
 *      Author: tombr
 */

#include "BernoulliSampler.h"

#include <cstdlib>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(BernoulliSampler)

  ReflectableBase(workflow::DefaultWorkflowElement<BernoulliSampler>)

  WorkflowProperty(Parameters, Input("P"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(SampleCount)
  WorkflowProperty(Data, Output("D"))

EndPropertyDefinitions

BernoulliSampler::BernoulliSampler() : _SampleCount(1) {
  setLabel("BerSampler");
}

void BernoulliSampler::update(workflow::IProgressMonitor* monitor) const {
  std::vector<float>& parameters = *getParameters();
  const size_t featureCount = parameters.size();
  const size_t count = featureCount * getSampleCount();
  boost::shared_ptr<std::vector<float> > data(new std::vector<float>(count));

  for (size_t i = 0; i < data->size(); ++i) {
    data->at(i) = (float)rand() / (float)RAND_MAX < parameters[i % featureCount];
  }

  newState->setData(data);
}

} /* namespace ml */
} /* namespace gapputils */
