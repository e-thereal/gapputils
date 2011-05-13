/*
 * DefaultWorkflowElement.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef DEFAULTWORKFLOWELEMENT_H_
#define DEFAULTWORKFLOWELEMENT_H_

#include "WorkflowElement.h"

namespace gapputils {

namespace workflow {

class DefaultWorkflowElement : public WorkflowElement {

  InitReflectableClass(DefaultWorkflowElement)

public:
  DefaultWorkflowElement();
  virtual ~DefaultWorkflowElement();

  virtual void execute(IProgressMonitor* monitor) const { }
  virtual void writeResults() { }
};

}

}

#endif /* DEFAULTWORKFLOWELEMENT_H_ */
