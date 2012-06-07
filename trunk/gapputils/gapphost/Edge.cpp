#include "Edge.h"

#include <capputils/EventHandler.h>
#include <capputils/ObservableClass.h>
#include <capputils/VolatileAttribute.h>

#include "Node.h"

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Edge)

  DefineProperty(OutputNode)
  DefineProperty(OutputProperty)
  DefineProperty(OutputNodePtr, Volatile())

  DefineProperty(InputNode)
  DefineProperty(InputProperty)
  DefineProperty(InputNodePtr, Volatile())
  DefineProperty(CableItem, Volatile())

EndPropertyDefinitions

Edge::Edge(void) : _OutputNodePtr(0), _InputNodePtr(0), _CableItem(0), handler(this, &Edge::changedHandler)
{
}

Edge::~Edge(void)
{
  if (getOutputNodePtr()) {
    capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(getOutputNodePtr()->getModule());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
  }
}

bool Edge::activate(Node* outputNode, Node* inputNode) {
  // TODO: re-think how to activate an edge. How do I activate an edge that connects
  //       a node with an interface node of a workflow?

  // Get property IDs and use them for the rest.

  setOutputNodePtr(outputNode);
  setInputNodePtr(inputNode);

  if (!inputNode->getModule()->getPropertyIndex(inputId, getInputProperty()) ||
      !outputNode->getModule()->getPropertyIndex(outputId, getOutputProperty()))
  {
    return false;
  }

  capputils::reflection::IClassProperty* inProp = inputNode->getModule()->getProperties()[inputId];
  capputils::reflection::IClassProperty* outProp = outputNode->getModule()->getProperties()[outputId];

  if (!Edge::areCompatible(outputNode, outputId, inputNode, inputId))
    return false;

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputNode->getModule());
  if (observable) {
    observable->Changed.connect(handler);
  }

  if (inProp && outProp) {
    inProp->setValue(*inputNode->getModule(), *outputNode->getModule(), outProp);
  }

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

  Node* inputNode = getInputNodePtr();
  Node* outputNode = getOutputNodePtr();

  if (!inputNode || !outputNode)
    return;

  capputils::reflection::IClassProperty* inProp = inputNode->getModule()->getProperties()[inputId];
  capputils::reflection::IClassProperty* outProp = outputNode->getModule()->getProperties()[outputId];
  if (inProp && outProp) {
    inProp->setValue(*inputNode->getModule(), *outputNode->getModule(), outProp);
  }
}

}

}
