/*
 * OutputsItem.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "OutputsItem.h"

#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>

using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

OutputsItem::OutputsItem(Node* node, Workbench *bench) : ToolItem(node, bench) {
  deletable = false;
  updateConnections();
  updateSize();
}

OutputsItem::~OutputsItem() {
}

std::string OutputsItem::getLabel() const {
  return "Outputs";
}

void OutputsItem::updateConnections() {
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
    if (properties[i]->getAttribute<OutputAttribute>()) {
      string label = properties[i]->getName();
      if ((shortName = properties[i]->getAttribute<ShortNameAttribute>()))
        label = shortName->getName();
      inputs.push_back(new ToolConnection(label.c_str(), ToolConnection::Input, this, properties[i]));
    }
  }
}

}
