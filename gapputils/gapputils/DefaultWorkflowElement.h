/*
 * DefaultWorkflowElement.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_
#define GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Logbook.h>
#include <capputils/Verifier.h>

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/DummyAttribute.h>
#include <capputils/attributes/NotNullAttribute.h>
#include <capputils/attributes/NotEmptyAttribute.h>
#include <capputils/attributes/NotEqualAttribute.h>
#include <capputils/attributes/EnumeratorAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>
#include <gapputils/attributes/ReadOnlyAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/NoParameterAttribute.h>
#include <gapputils/attributes/LabelAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/FileExistsAttribute.h>
#include <capputils/attributes/FlagAttribute.h>

#include <capputils/attributes/EnumerableAttribute.h>
#include <capputils/attributes/FromEnumerableAttribute.h>
#include <capputils/attributes/ToEnumerableAttribute.h>

namespace gapputils {

namespace workflow {

/** Results are only written when update is overloaded */

template<class T>
class DefaultWorkflowElement : public WorkflowElement {
protected:
  mutable boost::shared_ptr<T> newState;

private:
  mutable bool writeEnabled;

public:
  DefaultWorkflowElement() : writeEnabled(true) { }

  virtual ~DefaultWorkflowElement() { }

  virtual void execute(IProgressMonitor* monitor) const {
    capputils::Logbook& dlog = getLogbook();
    if (!newState)
      newState = boost::make_shared<T>();

    if (!capputils::Verifier::Valid(*this, dlog)) {
      dlog(capputils::Severity::Warning) << "Invalid arguments. Aborting!";
      return;
    }

    update(monitor);
  }

  virtual void reset() {
    newState = boost::make_shared<T>();
  }

  virtual void writeResults() {
    if (!newState)
      return;

    if (writeEnabled) {
      std::vector<capputils::reflection::IClassProperty*>& properties = getProperties();
      for (unsigned i = 0; i < properties.size(); ++i) {
        capputils::reflection::IClassProperty* prop = properties[i];
        if (prop->getAttribute<capputils::attributes::NoParameterAttribute>() &&
            !prop->getAttribute<gapputils::attributes::LabelAttribute>())
        {
          prop->setValue(*this, *newState, prop);
        }
      }
    } else {
      std::vector<capputils::reflection::IClassProperty*>& properties = getProperties();
      for (unsigned i = 0; i < properties.size(); ++i) {
        capputils::reflection::IClassProperty* prop = properties[i];
        prop->setValue(*this, *this, prop);
      }
    }
  }

protected:
  virtual void update(IProgressMonitor* monitor) const {
    CAPPUTILS_UNUSED(monitor);
    writeEnabled = false;
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
    ::capputils::reflection::IClassProperty* prop = new ::capputils::reflection::ClassProperty<Type>(#name, _##name, ClassType ::get##name, ClassType ::set##name, ClassType ::reset##name, __VA_ARGS__, capputils::attributes::Observe(Id), NULL); \
    properties.push_back(prop); \
    if (is_pointer<Type>::value) { \
      if (!prop->getAttribute<capputils::attributes::IReflectableAttribute>()) \
        properties[properties.size()-1]->addAttribute(new capputils::attributes::VolatileAttribute()); \
      properties[properties.size()-1]->addAttribute(new gapputils::attributes::ReadOnlyAttribute()); \
    } \
    addressbook[#name] = (char*)&_##name - (char*)this; \
    CAPPUTILS_UNUSED(Id); \
  }

#else /* !defined(_MSC_VER) */

#define WorkflowProperty(name, arguments...) \
{ \
  typedef TYPE_OF(name) Type; \
  const unsigned Id = properties.size(); \
  ::capputils::reflection::IClassProperty* prop = new ::capputils::reflection::ClassProperty<Type>(#name, _##name, ClassType ::get##name, ClassType ::set##name, ClassType ::reset##name, ##arguments, capputils::attributes::Observe(Id), NULL); \
  properties.push_back(prop); \
  if (is_pointer<Type>::value) { \
    if (!prop->getAttribute<capputils::attributes::IReflectableAttribute>()) \
      properties[properties.size()-1]->addAttribute(new capputils::attributes::VolatileAttribute()); \
    /*if (!prop->getAttribute<capputils::attributes::FromEnumerableAttribute>())*/ \
    properties[properties.size()-1]->addAttribute(new gapputils::attributes::ReadOnlyAttribute()); \
  } \
  addressbook[#name] = (char*)&_##name - (char*)this; \
  CAPPUTILS_UNUSED(Id); \
}

#endif /* defined(_MSC_VER) */

#endif /* GAPPUTLIS_WORKFLOW_DEFAULTWORKFLOWELEMENT_H_ */
