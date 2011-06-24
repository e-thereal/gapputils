/*
 * GlobalProperty.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "GlobalProperty.h"

#include <capputils/VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(GlobalProperty)

  DefineProperty(Name)
  DefineProperty(ModuleUuid)
  DefineProperty(PropertyName)
  DefineProperty(NodePtr, Volatile())
  DefineProperty(PropertyId, Volatile())
  DefineProperty(Edges, Volatile())

EndPropertyDefinitions

GlobalProperty::GlobalProperty() : _NodePtr(0), _PropertyId(-1) {
  _Edges = new std::vector<Edge*>();
}

GlobalProperty::~GlobalProperty() {
  delete _Edges;
}

capputils::reflection::IClassProperty* GlobalProperty::getProperty() {
  Node* node = getNodePtr();
  if (node) {
    capputils::reflection::ReflectableClass* object = node->getModule();
    if (object) {
      return object->getProperties()[getPropertyId()];
    }
  }
  return 0;
}

}

}
