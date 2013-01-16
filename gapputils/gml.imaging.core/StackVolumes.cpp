/*
 * StackVolumes.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#include "StackVolumes.h"

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(StackVolumes)

  ReflectableBase(DefaultWorkflowElement<StackVolumes>)

  WorkflowProperty(Volumes1, Input("Vols1"))
  WorkflowProperty(Volumes2, Input("Vols2"))
  WorkflowProperty(Volume1, Input("Vol1"))
  WorkflowProperty(Volume2, Input("Vol2"))
  WorkflowProperty(Output, Output("Vols"))

EndPropertyDefinitions

StackVolumes::StackVolumes() {
  setLabel("Stack");
}

void StackVolumes::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > output(new std::vector<boost::shared_ptr<image_t> >());

  if (getVolumes1()) {
    std::vector<boost::shared_ptr<image_t> >& inputs = *getVolumes1();
    for (size_t i = 0; i < inputs.size(); ++i)
      output->push_back(inputs[i]);
  }

  if (getVolumes2()) {
    std::vector<boost::shared_ptr<image_t> >& inputs = *getVolumes2();
    for (size_t i = 0; i < inputs.size(); ++i)
      output->push_back(inputs[i]);
  }

  if (getVolume1())
    output->push_back(getVolume1());

  if (getVolume2())
    output->push_back(getVolume2());

  if (!output->size()) {
    dlog(Severity::Warning) << "No volumes given. Output stack is empty.";
  }

  newState->setOutput(output);
}


} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
