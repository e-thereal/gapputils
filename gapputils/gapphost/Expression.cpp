/*
 * Expression.cpp
 *
 *  Created on: Jan 27, 2012
 *      Author: tombr
 */

#include "Expression.h"

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>

#include <cassert>
#include <sstream>
#include <iostream>

#include "Node.h"
#include "GlobalProperty.h"
#include "Workflow.h"
#include "PropertyReference.h"

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Expression)
  using namespace capputils::attributes;

  DefineProperty(Expression)
  DefineProperty(PropertyName, Description("Name of the property the expression is bound to."))
  DefineProperty(Node, Volatile())

EndPropertyDefinitions

Expression::Expression() : handler(this, &Expression::changedHandler) {

}

Expression::~Expression() {
  disconnectAll();
}

std::string Expression::evaluate() const {
  assert(!getNode().expired());
  assert(!getNode().lock()->getWorkflow().expired());

  boost::shared_ptr<Workflow> workflow = getNode().lock()->getWorkflow().lock();
  std::stringstream input(getExpression());
  std::stringstream output;

  char ch;
  input.read(&ch, 1);
//  input >> ch;
  for(input.read(&ch, 1); !input.eof(); input.read(&ch, 1)) {
    if (ch == '$') {
      input.read(&ch, 1);
      if (ch == '(') {
        std::stringstream propertyName;
        for (input.read(&ch, 1); !input.eof() && ch != ')'; input.read(&ch, 1))
          propertyName << ch;
        boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(propertyName.str());
        if (gprop) {
          boost::shared_ptr<PropertyReference> sref = PropertyReference::TryCreate(
              workflow, gprop->getModuleUuid(), gprop->getPropertyId());
          if (sref) {
            PropertyReference& ref = *sref;
            output << ref.getProperty()->getStringValue(*ref.getObject());
          } else {
            output << "{" << propertyName.str() << " not found}";
          }
        } else {
          output << "{" << propertyName.str() << " not found}";
        }
      } else {
        output << '$' << ch;
      }
    } else {
      output << ch;
    }
  }

  return output.str();
}

int getPropertyPos(const capputils::reflection::ReflectableClass& object,
    const capputils::reflection::IClassProperty* prop)
{
  std::vector<capputils::reflection::IClassProperty*>& props = object.getProperties();
  for (unsigned i = 0; i < props.size(); ++i) {
    if (props[i] == prop)
      return i;
  }
  assert(0);
  return 0;
}

bool Expression::resume() {
  assert(!getNode().expired());
  assert(!getNode().lock()->getWorkflow().expired());

  disconnectAll();

  // Find global properties and add event handler to them

  boost::shared_ptr<Workflow> workflow = getNode().lock()->getWorkflow().lock();
  std::stringstream input(getExpression());

  char ch;
  input >> ch;
  for(input >> ch; !input.eof(); input >> ch) {
    if (ch == '$') {
      input >> ch;
      if (ch == '(') {
        std::stringstream propertyName;
        for (input >> ch; !input.eof() && ch != ')'; input >> ch)
          propertyName << ch;
        boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(propertyName.str());
        if (gprop) {
          boost::shared_ptr<PropertyReference> sref = PropertyReference::TryCreate(
              workflow, gprop->getModuleUuid(), gprop->getPropertyId());
          if (sref) {
            PropertyReference& ref = *sref;
            capputils::ObservableClass* observable =
                dynamic_cast<capputils::ObservableClass*>(ref.getObject());
            if (observable) {
              // don't link twice for the same global property
              if (globalProperties.find(gprop) == globalProperties.end()) {
                observable->Changed.connect(handler);
                observedProperties.insert(std::pair<capputils::ObservableClass*, int>(observable,
                    getPropertyPos(*ref.getObject(), ref.getProperty())));
                gprop->getExpressions()->push_back(shared_from_this());
                globalProperties.insert(gprop);
              }
            }
          }
        }
      }
    }
  }

  // Trigger the global properties changed event
  workflow->setGlobalProperties(workflow->getGlobalProperties());

  ReflectableClass* object = getNode().lock()->getModule().get();
  assert(object);
  if (!object->hasProperty(getPropertyName()))
    return false;

  object->findProperty(getPropertyName())->setStringValue(*object, evaluate());
  return true;
}

void Expression::disconnect(boost::shared_ptr<GlobalProperty> gprop) {
  // Mean I'll tell the global property that I'm gone now and I remove my event handler
  // and the pair
  assert(gprop);

  if (globalProperties.find(gprop) != globalProperties.end()) {
    for (unsigned i = 0; i < gprop->getExpressions()->size(); ++i) {
      // If this function is called from the destructor, the expression pointer has already expired and a simple comparison with this would fail
      // Hence, also delete expressions that have expired
      if (gprop->getExpressions()->at(i).expired() || gprop->getExpressions()->at(i).lock().get() == this) {
        gprop->getExpressions()->erase(gprop->getExpressions()->begin() + i);
        --i;
      }
    }

    // If the node is about to be deleted, the weak pointer is already invalid so check for
    // that case before doing anything else
    if (!getNode().expired()) {
      boost::shared_ptr<Workflow> workflow = getNode().lock()->getWorkflow().lock();
      PropertyReference ref(workflow, gprop->getModuleUuid(), gprop->getPropertyId());

      capputils::ObservableClass* observable =
          dynamic_cast<capputils::ObservableClass*>(ref.getObject());

      if (observable) {
        observable->Changed.disconnect(handler);
        std::pair<capputils::ObservableClass*, int> pair(observable,
            getPropertyPos(*ref.getObject(), ref.getProperty()));
        std::set<std::pair<capputils::ObservableClass*, int> >::iterator iPair = observedProperties.find(pair);
        if (iPair != observedProperties.end())
          observedProperties.erase(iPair);
      }

      // Trigger the global properties have changed event
      workflow->setGlobalProperties(workflow->getGlobalProperties());
    }
    globalProperties.erase(gprop);
  }
}

void Expression::disconnectAll() {
  while (globalProperties.size()) {
    disconnect(globalProperties.begin()->lock());
  }
}

void Expression::changedHandler(capputils::ObservableClass* sender, int eventId) {
  using namespace capputils::reflection;

  assert(!getNode().expired());
  assert(getNode().lock()->getModule());

  std::pair<capputils::ObservableClass*, int> senderEventId(sender, eventId);

  if (observedProperties.find(senderEventId) != observedProperties.end()) {
    ReflectableClass* object = getNode().lock()->getModule().get();
    IClassProperty* property = object->findProperty(getPropertyName());
    if (property) {
      property->setStringValue(*object, evaluate());
    }
  } else {
    //std::cout << "[Info] Change of unobserved property detected." << std::endl;
  }
}

} /* namespace workflow */

} /* namespace gapputils */
