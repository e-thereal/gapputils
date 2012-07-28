/*
 * DefaultWorkflowElement.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_
#define GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_

#include "WorkflowElement.h"

#include <capputils/NotNullAttribute.h>
#include <capputils/NotEmptyAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/Logbook.h>
#include <capputils/NoParameterAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>

#include <capputils/EnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>

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
  virtual void update(IProgressMonitor* monitor) const {
    CAPPUTILS_UNUSED(monitor);
  }
};

}

}

template<class T>
class is_pointer {
public:
  enum Result { value = 0 };
};

template<class T>
class is_pointer<T*> {
public:
  enum Result { value = 1 };
};

template<class T>
class is_pointer<boost::shared_ptr<T> > {
public:
  enum Result { value = 1 };
};

#if defined(_MSC_VER)

#define WorkflowProperty(name, ...) \
  { \
    typedef TYPE_OF(name) Type; \
    const unsigned Id = properties.size(); \
    properties.push_back(new ::capputils::reflection::ClassProperty<Type>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, capputils::attributes::Observe(Id), 0)); \
    if (is_pointer<Type>::value) { \
      properties[properties.size()-1]->addAttribute(new capputils::attributes::VolatileAttribute()); \
      properties[properties.size()-1]->addAttribute(new gapputils::attributes::ReadOnlyAttribute()); \
    } \
    CAPPUTILS_UNUSED(Id); \
  }

#else /* !defined(_MSC_VER) */

#define WorkflowProperty(name, arguments...) \
{ \
  typedef TYPE_OF(name) Type; \
  const unsigned Id = properties.size(); \
  ::capputils::reflection::IClassProperty* prop = new ::capputils::reflection::ClassProperty<Type>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, capputils::attributes::Observe(Id), 0); \
  properties.push_back(prop); \
  if (is_pointer<Type>::value) { \
    properties[properties.size()-1]->addAttribute(new capputils::attributes::VolatileAttribute()); \
    /*if (!prop->getAttribute<capputils::attributes::FromEnumerableAttribute>())*/ \
      properties[properties.size()-1]->addAttribute(new gapputils::attributes::ReadOnlyAttribute()); \
  } \
  CAPPUTILS_UNUSED(Id); \
}

#endif /* defined(_MSC_VER) */

#endif /* GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_ */
