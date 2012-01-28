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

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Expression)
  using namespace capputils::attributes;

  DefineProperty(Expression)
  DefineProperty(PropertyName, Description("Name of the property the expression is bound to."))
  DefineProperty(Node, Volatile())

EndPropertyDefinitions

Expression::Expression() : _Node(0), handler(this, &Expression::changedHandler) {

}

Expression::~Expression() {
  disconnectAll();
}

std::string Expression::evaluate() const {
  assert(getNode());
  assert(getNode()->getWorkflow());

  Workflow* workflow = getNode()->getWorkflow();
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
        GlobalProperty* gprop = workflow->getGlobalProperty(propertyName.str());
        if (gprop && gprop->getProperty() && gprop->getNodePtr() && gprop->getNodePtr()->getModule()) {
          output << gprop->getProperty()->getStringValue(*gprop->getNodePtr()->getModule());
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

void Expression::resume() {
  assert(getNode());
  assert(getNode()->getWorkflow());

  disconnectAll();

  // Find global properties and add event handler to them

  Workflow* workflow = getNode()->getWorkflow();
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
        GlobalProperty* gprop = workflow->getGlobalProperty(propertyName.str());
        if (gprop && gprop->getProperty() && gprop->getNodePtr() && gprop->getNodePtr()->getModule()) {
          capputils::ObservableClass* observable =
              dynamic_cast<capputils::ObservableClass*>(gprop->getNodePtr()->getModule());
          if (observable) {
            // don't link twice for the same global property
            if (globalProperties.find(gprop) == globalProperties.end()) {
              observable->Changed.connect(handler);
              observedProperties.insert(std::pair<capputils::ObservableClass*, int>(observable, gprop->getPropertyId()));
              gprop->getExpressions()->push_back(this);
              globalProperties.insert(gprop);
            }
          }
        }
      }
    }
  }
}

void Expression::disconnect(GlobalProperty* gprop) {
  // Mean I'll tell the global property that I'm gone now and I remove my event handler
  // and the pair
  assert(gprop);
  assert(gprop->getNodePtr());

  capputils::ObservableClass* observable =
      dynamic_cast<capputils::ObservableClass*>(gprop->getNodePtr()->getModule());

  if (observable) {
    if (globalProperties.find(gprop) != globalProperties.end()) {
      for (unsigned i = 0; i < gprop->getExpressions()->size(); ++i) {
        if (gprop->getExpressions()->at(i) == this)
          gprop->getExpressions()->erase(gprop->getExpressions()->begin() + i);
      }
      observable->Changed.disconnect(handler);
      std::pair<capputils::ObservableClass*, int> pair(observable, gprop->getPropertyId());
      std::set<std::pair<capputils::ObservableClass*, int> >::iterator iPair = observedProperties.find(pair);
      if (iPair != observedProperties.end())
        observedProperties.erase(iPair);

      globalProperties.erase(gprop);
    }
  }
}

void Expression::disconnectAll() {
  while (globalProperties.size()) {
    disconnect(*globalProperties.begin());
  }
}

void Expression::changedHandler(capputils::ObservableClass* sender, int eventId) {
  using namespace capputils::reflection;

  assert(getNode());
  assert(getNode()->getModule());

  std::pair<capputils::ObservableClass*, int> senderEventId(sender, eventId);

  if (observedProperties.find(senderEventId) != observedProperties.end()) {
    ReflectableClass* object = getNode()->getModule();
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
