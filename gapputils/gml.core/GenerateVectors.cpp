/*
 * GenerateVectors.cpp
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#include "GenerateVectors.h"

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(GenerateVectors)

  ReflectableBase(DefaultWorkflowElement<GenerateVectors>)

  WorkflowProperty(From, NotEmpty<Type>())
  WorkflowProperty(StepCount, NotEmpty<Type>())
  WorkflowProperty(To, NotEmpty<Type>())
  WorkflowProperty(Order, NotEmpty<Type>())
  WorkflowProperty(Vectors, Output("Vecs"))

EndPropertyDefinitions

GenerateVectors::GenerateVectors() {
  setLabel("Vectors");
}

void GenerateVectors::update(workflow::IProgressMonitor* monitor) const {
  capputils::Logbook& dlog = getLogbook();

  std::vector<double> from = getFrom();
  std::vector<int> stepCount = getStepCount();
  std::vector<double> to = getTo();
  std::vector<int> order = getOrder();

  size_t dim = from.size();

  if (dim != to.size() || dim != stepCount.size() || dim != order.size()) {
    dlog(capputils::Severity::Warning) << "From, stepCount, to, and order vector must have the same size! Aborting.";
    return;
  }

  std::vector<int> sortedOrder(order);
  std::sort(sortedOrder.begin(), sortedOrder.end());

  for (size_t i = 0; i < sortedOrder.size(); ++i) {
    if (sortedOrder[i] != (int)i) {
      dlog(capputils::Severity::Warning) << "Order must be a permutation. Aborting!";
      return;
    }
  }

  boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > outputs(
      new std::vector<boost::shared_ptr<std::vector<double> > >());


  std::vector<int> i(dim);
  std::fill(i.begin(), i.end(), 0);

  while(i[order[dim-1]] < stepCount[order[dim-1]]) {
    // create vector
    boost::shared_ptr<std::vector<double> > output(new std::vector<double>());
    for (size_t j = 0; j < dim; ++j) {
      if (stepCount[j] <= 1)
        output->push_back(from[j]);
      else
        output->push_back((double)i[j] * (to[j] - from[j]) / (double)(stepCount[j] - 1) + (double)from[j]);
//      std::cout << output->at(output->size() - 1) << " ";
    }
    outputs->push_back(output);
//    std::cout << std::endl;

    // increment i
    ++i[order[0]];
    for (size_t j = 0; j < dim - 1; ++j) {
      if (i[order[j]] >= stepCount[order[j]]) {
        i[order[j]] = 0;
        ++i[order[j+1]];
      }
    }
  }

  newState->setVectors(outputs);
}

} /* namespace core */

} /* namespace gml */
