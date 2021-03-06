/*
 * WorkflowInterface.h
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_WORKFLOWINTERFACE_H_
#define GAPPUTILS_WORKFLOWINTERFACE_H_

#include "gapputils.h"

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <capputils/TimedClass.h>

namespace gapputils {

namespace workflow {

class WorkflowInterface : public capputils::reflection::ReflectableClass,
                          public capputils::ObservableClass
{

  InitReflectableClass(WorkflowInterface)

  Property(Label, std::string)

public:
  static int LabelId;

public:
  WorkflowInterface();
  virtual ~WorkflowInterface();
};

}

}

#define WfiUpdateTimestamp \
  { \
  if (hasProperty("SetOnCompilation")) { \
    capputils::attributes::TimeStampAttribute* timeStamp = findProperty("SetOnCompilation")->getAttribute<capputils::attributes::TimeStampAttribute>(); \
    if (timeStamp) \
      timeStamp->setTime(*this, __DATE__" "__TIME__); \
  } \
  }

#endif /* GAPPUTILS_WORKFLOWINTERFACE_H_ */
