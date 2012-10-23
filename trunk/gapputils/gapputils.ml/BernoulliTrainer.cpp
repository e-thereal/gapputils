/*
 * BernoulliTrainer.cpp
 *
 *  Created on: Oct 22, 2012
 *      Author: tombr
 */

#include "BernoulliTrainer.h"

#include <algorithm>

#include <boost/lambda/lambda.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(BernoulliTrainer)

  ReflectableBase(workflow::DefaultWorkflowElement<BernoulliTrainer>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FeatureCount, NotEqual(0))
  WorkflowProperty(Parameters, Output("P"))

EndPropertyDefinitions

BernoulliTrainer::BernoulliTrainer() : _FeatureCount(0) {
  setLabel("BerTrain");
}

BernoulliTrainer::~BernoulliTrainer() {
}

void BernoulliTrainer::update(workflow::IProgressMonitor* monitor) const {
  using namespace boost::lambda;

  const size_t featureCount = getFeatureCount();
  boost::shared_ptr<std::vector<float> > parameters(new std::vector<float>(featureCount));
  std::vector<float>& data = *getTrainingSet();
  const size_t sampleCount = data.size() / featureCount;

  std::fill(parameters->begin(), parameters->end(), 0.f);
  for (size_t offset = 0; offset < data.size(); offset += featureCount) {
    std::transform(parameters->begin(), parameters->end(), data.begin() + offset, parameters->begin(),
        _1 + _2);
  }
  std::transform(parameters->begin(), parameters->end(), parameters->begin(), _1 / (float)sampleCount);
  newState->setParameters(parameters);
}

} /* namespace ml */
} /* namespace gapputils */
