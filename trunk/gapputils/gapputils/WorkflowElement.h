/*
 * WorkflowElement.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_WORKFLOWELEMENT_H_
#define GAPPUTILS_WORKFLOWELEMENT_H_

#include <gapputils/gapputils.h>

#include <capputils/reflection/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <capputils/TimedClass.h>

#include <gapputils/IProgressMonitor.h>
#include <gapputils/IGapphostInterface.h>

namespace capputils {

class Logbook;

}

namespace gapputils {

namespace workflow {

class WorkflowElement : public capputils::reflection::ReflectableClass,
                        public capputils::ObservableClass
{

  InitAbstractReflectableClass(WorkflowElement)

  Property(Label, std::string)

public:
  static int labelId;

private:
  boost::shared_ptr<capputils::Logbook> logbook;
  boost::shared_ptr<IGapphostInterface> hostInterface;
  bool atomicWorkflow;

public:
  WorkflowElement();

  capputils::Logbook& getLogbook() const;
  boost::shared_ptr<IGapphostInterface> getHostInterface() const;
  void setHostInterface(const boost::shared_ptr<IGapphostInterface>& interface);

  bool getAtomicWorkflow() const;
  void setAtomicWorkflow(bool atomic);

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

  /**
   * \brief Called by the WorkflowUpdater when a node reset is performed
   *
   * Overload this method to free all temporary memory used by the module.
   */
  virtual void reset() { }
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
