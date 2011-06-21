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

void Edge::activate(Node* outputNode, Node* inputNode) {
  // Get property IDs and use them for the rest.

  setOutputNodePtr(outputNode);
  setInputNodePtr(inputNode);

  if (!inputNode->getModule()->getPropertyIndex(inputId, getInputProperty()) ||
      !outputNode->getModule()->getPropertyIndex(outputId, getOutputProperty()))
  {
    return;
  }

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputNode->getModule());
  if (observable) {
    observable->Changed.connect(handler);
  }

  capputils::reflection::IClassProperty* inProp = inputNode->getModule()->getProperties()[inputId];
  capputils::reflection::IClassProperty* outProp = outputNode->getModule()->getProperties()[outputId];
  if (inProp && outProp) {
    inProp->setValue(*inputNode->getModule(), *outputNode->getModule(), outProp);
  }
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
