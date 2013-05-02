/*
 * GlobalEdge.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "GlobalEdge.h"

#include <capputils/EventHandler.h>
#include <capputils/ObservableClass.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/MergeAttribute.h>

#include <capputils/Logbook.h>

#include "Node.h"
#include "PropertyReference.h"
#include "LogbookModel.h"

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(GlobalEdge)

  ReflectableBase(Edge)
  DefineProperty(GlobalProperty)

EndPropertyDefinitions

GlobalEdge::GlobalEdge() : Edge(), handler(this, &GlobalEdge::changedHandler), inputId(-1) { }

GlobalEdge::~GlobalEdge() {
  PropertyReference* inputRef = getInputReference().get();
  if (inputRef) {
    capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(inputRef->getObject());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
  }
}

bool GlobalEdge::activate(boost::shared_ptr<Node> outputNode, boost::shared_ptr<Node> inputNode) {
  // TODO: re-think how to activate an edge. How do I activate an edge that connects
  //       a node with an interface node of a workflow?

  // Get property IDs and use them for the rest.

  if (!this->Edge::activate(outputNode, inputNode))
    return false;

  if (!inputNode || inputNode->getWorkflow().expired())
    return false;

  boost::shared_ptr<PropertyReference> inputRef = PropertyReference::TryCreate(inputNode->getWorkflow().lock(), inputNode->getUuid(), getInputProperty());
  if (!inputRef)
    return false;

  std::vector<capputils::reflection::IClassProperty*>& properties = inputRef->getObject()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i] == inputRef->getProperty()) {
      inputId = i;
      break;
    }
  }

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(inputRef->getObject());
  if (observable) {
    observable->Changed.connect(handler);
  }

  return true;
}

void GlobalEdge::changedHandler(capputils::ObservableClass*, int eventId) {
  // check for the right property ID

  if (eventId == (int)inputId) {
    PropertyReference* inputRef = getInputReference().get();
    PropertyReference* outputRef = getOutputReference().get();

    if (!inputRef || !outputRef)
      return;

    capputils::reflection::IClassProperty* inProp = inputRef->getProperty();
    capputils::reflection::IClassProperty* outProp = outputRef->getProperty();
    if (inProp && outProp && outProp->compare(*outputRef->getObject(), *inputRef->getObject(), inProp)) {
      outProp->setValue(*outputRef->getObject(), *inputRef->getObject(), inProp);
    }
  }
}

}

}
