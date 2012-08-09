/*
 * Expression.cpp
 *
 *  Created on: Jan 27, 2012
 *      Author: tombr
 */

#include "Expression.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/VolatileAttribute.h>

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
          PropertyReference ref(workflow, gprop->getModuleUuid(), gprop->getPropertyId());
          output << ref.getProperty()->getStringValue(*ref.getObject());
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

void Expression::resume() {
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
          PropertyReference ref(workflow, gprop->getModuleUuid(), gprop->getPropertyId());
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

  ReflectableClass* object = getNode().lock()->getModule().get();
  assert(object);
  object->findProperty(getPropertyName())->setStringValue(*object, evaluate());
}

void Expression::disconnect(boost::shared_ptr<GlobalProperty> gprop) {
  // Mean I'll tell the global property that I'm gone now and I remove my event handler
  // and the pair
  assert(gprop);

  boost::shared_ptr<Workflow> workflow = getNode().lock()->getWorkflow().lock();
  PropertyReference ref(workflow, gprop->getModuleUuid(), gprop->getPropertyId());

  capputils::ObservableClass* observable =
      dynamic_cast<capputils::ObservableClass*>(ref.getObject());

  if (observable) {
    if (globalProperties.find(gprop) != globalProperties.end()) {
      for (unsigned i = 0; i < gprop->getExpressions()->size(); ++i) {
        if (gprop->getExpressions()->at(i).lock().get() == this)
          gprop->getExpressions()->erase(gprop->getExpressions()->begin() + i);
      }
      observable->Changed.disconnect(handler);
      std::pair<capputils::ObservableClass*, int> pair(observable,
          getPropertyPos(*ref.getObject(), ref.getProperty()));
      std::set<std::pair<capputils::ObservableClass*, int> >::iterator iPair = observedProperties.find(pair);
      if (iPair != observedProperties.end())
        observedProperties.erase(iPair);

      globalProperties.erase(gprop);
    }
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
