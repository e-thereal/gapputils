#include "Edge.h"

#include <capputils/attributes/VolatileAttribute.h>
#include <capputils/attributes/MergeAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>

#include <capputils/Logbook.h>

#include "Node.h"
#include "PropertyReference.h"
#include "LogbookModel.h"

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

int Edge::positionId;

BeginPropertyDefinitions(Edge)

  DefineProperty(OutputNode)
  DefineProperty(OutputProperty)
  DefineProperty(OutputNodePtr, Volatile())
  DefineProperty(OutputReference, Volatile())

  DefineProperty(InputNode)
  DefineProperty(InputProperty)
  DefineProperty(InputNodePtr, Volatile())
  DefineProperty(InputReference, Volatile())
  DefineProperty(InputPosition, Observe(positionId = Id), Volatile())

  DefineProperty(CableItem, Volatile())

EndPropertyDefinitions

Edge::Edge(void) : _InputPosition(0), _CableItem(0), handler(this, &Edge::changedHandler), outputId(-1) {
  Changed.connect(handler);
}

Edge::~Edge(void) {
  PropertyReference* outputRef = getOutputReference().get();
  if (outputRef) {
    capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputRef->getObject());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
  }
}

bool Edge::activate(boost::shared_ptr<Node> outputNode, boost::shared_ptr<Node> inputNode) {
  // TODO: re-think how to activate an edge. How do I activate an edge that connects
  //       a node with an interface node of a workflow?

  // Get property IDs and use them for the rest.

  setOutputNodePtr(outputNode);
  setInputNodePtr(inputNode);

  if (!outputNode || outputNode->getWorkflow().expired())
    return false;

  boost::shared_ptr<PropertyReference> outputRef = PropertyReference::TryCreate(outputNode->getWorkflow().lock(), outputNode->getUuid(), getOutputProperty());
  if (!outputRef)
    return false;

  std::vector<capputils::reflection::IClassProperty*>& properties = outputRef->getObject()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i] == outputRef->getProperty()) {
      outputId = i;
      break;
    }
  }

  boost::shared_ptr<PropertyReference> inputRef = PropertyReference::TryCreate(inputNode->getWorkflow().lock(), inputNode->getUuid(), getInputProperty());
  if (!inputRef)
    return false;

  if (!Edge::areCompatible(outputRef->getProperty(), inputRef->getProperty())) {
    return false;
  }

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(outputRef->getObject());
  if (observable) {
    observable->Changed.connect(handler);
  }

  if (inputRef->getProperty() && outputRef->getProperty()) {
    IMergeAttribute* merge = inputRef->getProperty()->getAttribute<IMergeAttribute>();
    if (merge) {
      merge->setValue(*inputRef->getObject(), inputRef->getProperty(), getInputPosition(),
          *outputRef->getObject(), outputRef->getProperty());
    } else {
      inputRef->getProperty()->setValue(*inputRef->getObject(), *outputRef->getObject(), outputRef->getProperty());
    }
  }

  setInputReference(inputRef);
  setOutputReference(outputRef);

  return true;
}

bool Edge::areCompatible(const capputils::reflection::IClassProperty* outProp,
      const capputils::reflection::IClassProperty* inProp)
{
  IMergeAttribute* mergeAttribute = inProp->getAttribute<IMergeAttribute>();
  if (mergeAttribute)
    return outProp->getType() == mergeAttribute->getValueType();
  else
    return outProp->getType() == inProp->getType();
}

void Edge::changedHandler(capputils::ObservableClass* object, int eventId) {
  // check for the right property ID

  if ((object != this && eventId == (int)outputId) || (object == this && eventId == positionId)) {
    PropertyReference* inputRef = getInputReference().get();
    PropertyReference* outputRef = getOutputReference().get();

    if (!inputRef || !outputRef)
      return;

    capputils::reflection::IClassProperty* inProp = inputRef->getProperty();
    capputils::reflection::IClassProperty* outProp = outputRef->getProperty();

    if (inProp && outProp) {
      IMergeAttribute* merge = inProp->getAttribute<IMergeAttribute>();
      if (merge) {
        merge->setValue(*inputRef->getObject(), inProp, getInputPosition(), *outputRef->getObject(), outProp);
      } else {
        inProp->setValue(*inputRef->getObject(), *outputRef->getObject(), outProp);
      }
    }
  }
}

}

}
