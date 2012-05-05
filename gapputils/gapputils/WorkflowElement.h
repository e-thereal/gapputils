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
#include "IGapphostInterface.h"

namespace gapputils {

namespace workflow {

class WorkflowElement : public capputils::reflection::ReflectableClass,
                        public capputils::ObservableClass
{

  InitAbstractReflectableClass(WorkflowElement)

  Property(Label, std::string)
  Property(HostInterface, boost::shared_ptr<IGapphostInterface>)

public:
  static int labelId;

public:
  WorkflowElement();

  virtual void execute(IProgressMonitor* monitor) const = 0;
  virtual void writeResults() = 0;

  /**
   * \brief Called when an item is double clicked
   * 
   * Overload it to handle double clicks. This is supposed to be used to show
   * additional dialogs. Be creative. ;)
   */
  virtual void show() { }

  /**
   * \brief Called when the workflow is resumed
   *
   * Overload this method if a module needs to be initialized after all
   * properties have been read.
   */
  virtual void resume() { }
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
