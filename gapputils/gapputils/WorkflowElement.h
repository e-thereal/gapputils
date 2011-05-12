/*
 * WorkflowElement.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef WORKFLOWELEMENT_H_
#define WORKFLOWELEMENT_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

#include "IProgressMonitor.h"

namespace gapputils {

namespace workflow {

class WorkflowElement : public capputils::reflection::ReflectableClass,
                        public capputils::ObservableClass {

InitAbstractReflectableClass(WorkflowElement)

Property(Label, std::string)

public:
  WorkflowElement();

  virtual void execute(IProgressMonitor* monitor) const = 0;
  virtual void writeResults() = 0;
};

}

}

#endif /* WORKFLOWELEMENT_H_ */
