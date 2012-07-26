/*
 * DefaultWorkflowElement.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_
#define GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_

#include "WorkflowElement.h"

#include <capputils/Verifier.h>
#include <capputils/Logbook.h>
#include <capputils/NoParameterAttribute.h>
#include <gapputils/LabelAttribute.h>

// TODO: Introduce WorkflowProperty macro. This macro automatically sets certain attributes

namespace gapputils {

namespace workflow {

template<class T>
class DefaultWorkflowElement : public WorkflowElement {
protected:
  mutable T* newState;

public:
  DefaultWorkflowElement() : newState(0) { }

  virtual ~DefaultWorkflowElement() {
    if (newState)
      delete newState;
    newState = 0;
  }

  virtual void execute(IProgressMonitor* monitor) const {
    capputils::Logbook& dlog = getLogbook();
    if (!newState)
      newState = new T();

    if (!capputils::Verifier::Valid(*this, dlog)) {
      dlog(capputils::Severity::Warning) << "Invalid arguments. Aborting!";
      return;
    }

    update(monitor);
  }

  virtual void writeResults() {
    if (!newState)
      return;

    std::vector<capputils::reflection::IClassProperty*>& properties = getProperties();
    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      if (prop->getAttribute<capputils::attributes::NoParameterAttribute>() &&
          !prop->getAttribute<gapputils::attributes::LabelAttribute>())
      {
        prop->setValue(*this, *newState, prop);
      }
    }
  }

protected:
  virtual void update(IProgressMonitor* monitor) const { }
};

}

}

#endif /* GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_ */
