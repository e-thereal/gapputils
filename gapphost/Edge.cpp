#include "Edge.h"

#include <capputils/EventHandler.h>
#include <capputils/ObservableClass.h>
#include <capputils/VolatileAttribute.h>

#include "Node.h"
#include "PropertyReference.h"

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Edge)

  DefineProperty(OutputNode)
  DefineProperty(OutputProperty)
  DefineProperty(OutputNodePtr, Volatile())
  DefineProperty(OutputReference, Volatile())

  DefineProperty(InputNode)
  DefineProperty(InputProperty)
  DefineProperty(InputNodePtr, Volatile())
  DefineProperty(InputReference, Volatile())

  DefineProperty(CableItem, Volatile())

EndPropertyDefinitions

Edge::Edge(void)
 : _OutputNodePtr(0), _OutputReference(0), _InputNodePtr(0), _InputReference(0),
   _CableItem(0), handler(this, &Edge::changedHandler), outputId(-1)
{
}

Edge::~Edge(void)
{
  PropertyReference* inputRef = getInputReference();
  PropertyReference* outputRef = getOutputReference();
  if (outputRef) {
    capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputRef->getObject());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
    delete outputRef;
    setOutputReference(0);
  }

  if (inputRef) {
    delete inputRef;
    setInputReference(0);
  }
}

bool Edge::activate(Node* outputNode, Node* inputNode) {
  // TODO: re-think how to activate an edge. How do I activate an edge that connects
  //       a node with an interface node of a workflow?

  // Get property IDs and use them for the rest.

  setOutputNodePtr(outputNode);
  setInputNodePtr(inputNode);

  PropertyReference* outputRef = outputNode->getPropertyReference(getOutputProperty());
  if (!outputRef)
    return false;

  std::vector<capputils::reflection::IClassProperty*>& properties = outputRef->getObject()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i] == outputRef->getProperty()) {
      outputId = i;
      break;
    }
  }

  PropertyReference* inputRef = inputNode->getPropertyReference(getInputProperty());
  if (!inputRef)
    return false;

  if (!Edge::areCompatible(outputRef->getProperty(), inputRef->getProperty())) {
    delete inputRef;
    delete outputRef;
    return false;
  }

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputRef->getObject());
  if (observable) {
    observable->Changed.connect(handler);
  }

  if (inputRef->getProperty() && outputRef->getProperty()) {
    inputRef->getProperty()->setValue(*inputRef->getObject(), *outputRef->getObject(), outputRef->getProperty());
  }

  setInputReference(inputRef);
  setOutputReference(outputRef);

  return true;
}

bool Edge::areCompatible(const Node* outputNode, int outputId, const Node* inputNode, int inputId) {
  if (!outputNode || !outputNode->getModule() || !inputNode || !inputNode->getModule())
    return false;

  capputils::reflection::IClassProperty* inProp = inputNode->getModule()->getProperties()[inputId];
  capputils::reflection::IClassProperty* outProp = outputNode->getModule()->getProperties()[outputId];

  return Edge::areCompatible(outProp, inProp);
}

bool Edge::areCompatible(const capputils::reflection::IClassProperty* outProp,
      const capputils::reflection::IClassProperty* inProp)
{
  return outProp->getType() == inProp->getType();
}

void Edge::changedHandler(capputils::ObservableClass*, int eventId) {
  // check for the right property ID

  if (eventId != (int)outputId)
    return;

  PropertyReference* inputRef = getInputReference();
  PropertyReference* outputRef = getOutputReference();

  if (!inputRef || !outputRef)
    return;

  capputils::reflection::IClassProperty* inProp = inputRef->getProperty();
  capputils::reflection::IClassProperty* outProp = outputRef->getProperty();
  if (inProp && outProp) {
    inProp->setValue(*inputRef->getObject(), *outputRef->getObject(), outProp);
  }
}

}

}
