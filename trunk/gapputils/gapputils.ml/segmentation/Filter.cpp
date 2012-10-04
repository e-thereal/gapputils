/*
 * Trainer.cpp
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#include "Filter.h"

#include <capputils/PropertyMap.h>

namespace gapputils {
namespace ml {
namespace segmentation {

BeginPropertyDefinitions(Filter)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;

  ReflectableBase(workflow::DefaultWorkflowElement<Filter>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>())
  WorkflowProperty(Image, Input("Img"), NotNull<Type>())
  WorkflowProperty(Padded, Output("Pad"))
  WorkflowProperty(Centered, Output("Cent"))
  WorkflowProperty(Output, Output("Img"))

EndPropertyDefinitions

Filter::Filter() {
  setLabel("Filter");
}

  Filter::~Filter() {
}

void update_filter(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor);

void Filter::update(gapputils::workflow::IProgressMonitor* monitor) const {
  capputils::PropertyMap properties(*this);
  update_filter(properties, getLogbook(), monitor);
  newState->setPadded(properties.getValue<boost::shared_ptr<image_t> >("Padded"));
  newState->setCentered(properties.getValue<boost::shared_ptr<image_t> >("Centered"));
  newState->setOutput(properties.getValue<boost::shared_ptr<image_t> >("Output"));
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
