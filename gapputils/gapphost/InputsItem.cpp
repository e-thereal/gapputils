/*
 * InputsItem.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "InputsItem.h"

#include <InputAttribute.h>

using namespace capputils::reflection;
using namespace std;

namespace gapputils {

using namespace workflow;
using namespace attributes;

InputsItem::InputsItem(Node* node, Workbench *bench) : ToolItem(node, bench) {
  updateConnections();
  updateSize();
}

InputsItem::~InputsItem() {
}

std::string InputsItem::getLabel() const {
  return "Inputs";
}

void InputsItem::updateConnections() {
  for (unsigned i = 0; i < inputs.size(); ++i)
    delete inputs[i];
  for (unsigned i = 0; i < outputs.size(); ++i)
    delete outputs[i];
  inputs.clear();
  outputs.clear();

  ReflectableClass* object = node->getModule();
  vector<IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<InputAttribute>()) {
      outputs.push_back(new ToolConnection(properties[i]->getName().c_str(), ToolConnection::Output, this, properties[i]));
    }
  }
}

}
