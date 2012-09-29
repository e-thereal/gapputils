/*
 * Trainer.cpp
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#include "Trainer.h"

#include <capputils/PropertyMap.h>

namespace gapputils {
namespace ml {
namespace segmentation {

BeginPropertyDefinitions(Trainer)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;

  ReflectableBase(workflow::DefaultWorkflowElement<Trainer>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>())
  WorkflowProperty(Image, Input("Img"), NotNull<Type>())
  WorkflowProperty(Output, Output("Img"))

EndPropertyDefinitions

Trainer::Trainer() {
  setLabel("Trainer");
}

Trainer::~Trainer() {
}

void update_trainer(capputils::PropertyMap& properties, capputils::Logbook& logbook,
    workflow::IProgressMonitor* monitor);

void Trainer::update(gapputils::workflow::IProgressMonitor* monitor) const {
  capputils::PropertyMap properties(*this);
  update_trainer(properties, getLogbook(), monitor);
  newState->setOutput(properties.getValue<boost::shared_ptr<image_t> >("Output"));
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
