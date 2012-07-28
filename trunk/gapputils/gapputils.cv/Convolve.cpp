/*
 * Convolve.cpp
 *
 *  Created on: Jul 26, 2012
 *      Author: tombr
 */

#include "Convolve.h"

#include <capputils/PropertyMap.h>

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Convolve)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;

  ReflectableBase(workflow::DefaultWorkflowElement<Convolve>)

  WorkflowProperty(InputImage, Input("I"), NotNull<Type>());
  WorkflowProperty(Filter, Input("F"), NotNull<Type>());
  WorkflowProperty(Type, Enumerator<Type>());
  WorkflowProperty(OutputImage, Output(""));

EndPropertyDefinitions

Convolve::Convolve() {
  setLabel("Convolve");
}

void update_convolve(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor);

void Convolve::update(workflow::IProgressMonitor* monitor) const {
  capputils::PropertyMap properties(*this);
  update_convolve(properties, getLogbook(), monitor);
  newState->setOutputImage(properties.getValue<boost::shared_ptr<image_t> >("OutputImage"));
}

} /* namespace cv */
} /* namespace gapputils */
