/*
 * WorkflowElement.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef WORKFLOWELEMENT_H_
#define WORKFLOWELEMENT_H_

#include "gapputils.h"

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <capputils/TimedClass.h>

#include "IProgressMonitor.h"

namespace gapputils {

namespace workflow {

class WorkflowElement : public capputils::reflection::ReflectableClass,
                        public capputils::ObservableClass,
                        public capputils::TimedClass
{

  InitAbstractReflectableClass(WorkflowElement)

  Property(Label, std::string)
  Property(SetOnCompilation, int)

public:
  WorkflowElement();

  virtual void execute(IProgressMonitor* monitor) const = 0;
  virtual void writeResults() = 0;
};

}

}

#define WfeUpdateTimestamp \
  { \
  if (hasProperty("SetOnCompilation")) { \
    capputils::attributes::TimeStampAttribute* timeStamp = findProperty("SetOnCompilation")->getAttribute<capputils::attributes::TimeStampAttribute>(); \
    if (timeStamp) \
      timeStamp->setTime(*this, __DATE__" "__TIME__); \
  } \
  }

#endif /* WORKFLOWELEMENT_H_ */
