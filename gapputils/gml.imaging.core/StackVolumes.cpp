/*
 * StackVolumes.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#include "StackVolumes.h"

#include <capputils/MergeAttribute.h>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(StackVolumes)

  ReflectableBase(DefaultWorkflowElement<StackVolumes>)

  WorkflowProperty(Volumes, Input("Vs"), Merge<Type>())
  WorkflowProperty(Volume, Input("V"), Merge<Type>())
  WorkflowProperty(Output, Output("Vs"))

EndPropertyDefinitions

StackVolumes::StackVolumes() {
  setLabel("Stack");
}

void StackVolumes::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > output(new std::vector<boost::shared_ptr<image_t> >());

  if (getVolumes()) {
    for (size_t j = 0; j < getVolumes()->size(); ++j) {
      std::vector<boost::shared_ptr<image_t> >& inputs = *getVolumes()->at(j);
      for (size_t i = 0; i < inputs.size(); ++i)
        output->push_back(inputs[i]);
    }
  }

  if (getVolume()) {
    for (size_t i = 0; i < getVolume()->size(); ++i)
      output->push_back(getVolume()->at(i));
  }

  if (!output->size()) {
    dlog(Severity::Warning) << "No volumes given. Output stack is empty.";
  }

  newState->setOutput(output);
}


} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
