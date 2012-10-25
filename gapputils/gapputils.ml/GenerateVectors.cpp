/*
 * GenerateVectors.cpp
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#include "GenerateVectors.h"

#include <algorithm>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(GenerateVectors)

  ReflectableBase(workflow::DefaultWorkflowElement<GenerateVectors>)

  WorkflowProperty(Vectors, Output("Vecs"))
  WorkflowProperty(From, NotEmpty<Type>())
  WorkflowProperty(StepCount, NotEmpty<Type>())
  WorkflowProperty(To, NotEmpty<Type>())

EndPropertyDefinitions

GenerateVectors::GenerateVectors() {
  setLabel("Vectors");
}

GenerateVectors::~GenerateVectors() {
}

void GenerateVectors::update(workflow::IProgressMonitor* monitor) const {
  capputils::Logbook& dlog = getLogbook();

  std::vector<float> from = getFrom();
  std::vector<int> stepCount = getStepCount();
  std::vector<float> to = getTo();

  size_t dim = from.size();

  if (dim != to.size() || dim != stepCount.size()) {
    dlog(capputils::Severity::Warning) << "From, stepCount and to vector must have the same size! Aborting.";
    return;
  }

  boost::shared_ptr<std::vector<float> > output(new std::vector<float>);

  std::vector<int> i(dim);
  std::fill(i.begin(), i.end(), 0);

  while(i[dim-1] < stepCount[dim-1]) {
    // create vector
    for (size_t j = 0; j < dim; ++j) {
      if (stepCount[j] <= 1)
        output->push_back(from[j]);
      else
        output->push_back((float)i[j] * (to[j] - from[j]) / (float)(stepCount[j] - 1));
    }

    // increment i
    ++i[0];
    for (size_t j = 0; j < dim - 1; ++j) {
      if (i[j] >= stepCount[j]) {
        i[j] = 0;
        ++i[j + 1];
      }
    }
  }

  newState->setVectors(output);
}

} /* namespace ml */
} /* namespace gapputils */
