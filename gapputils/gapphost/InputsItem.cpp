/*
 * InputsItem.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "InputsItem.h"

#include <InputAttribute.h>
#include <ShortNameAttribute.h>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

InputsItem::InputsItem(Node* node, Workbench *bench) : ToolItem(node, bench) {
  deletable = false;
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
  ShortNameAttribute* shortName = 0;
  vector<IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<InputAttribute>()) {
      string label = properties[i]->getName();
      if ((shortName = properties[i]->getAttribute<ShortNameAttribute>()))
        label = shortName->getName();
      outputs.push_back(new MultiConnection(label.c_str(), ToolConnection::Output, this, properties[i]));
    }
  }
}

}
