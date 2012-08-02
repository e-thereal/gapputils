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
  DefineProperty(PropertyId)
//  DefineProperty(NodePtr, Volatile())
//  DefineProperty(PropertyId, Volatile())
  DefineProperty(Edges, Volatile())
  DefineProperty(Expressions, Volatile())

EndPropertyDefinitions

GlobalProperty::GlobalProperty()
 : _Expressions(new std::vector<Expression*>())
{
  _Edges = new std::vector<Edge*>();
}

GlobalProperty::~GlobalProperty() {
  delete _Edges;
}

//capputils::reflection::IClassProperty* GlobalProperty::getProperty() {
//  Node* node = getNodePtr();
//  if (node) {
//    capputils::reflection::ReflectableClass* object = node->getModule();
//    if (object) {
//      return object->getProperties()[getPropertyId()];
//    }
//  }
//  return 0;
//}

void GlobalProperty::addEdge(Edge* edge) {
  _Edges->push_back(edge);
}

void GlobalProperty::removeEdge(Edge* edge) {
  for (unsigned i = 0; i < _Edges->size(); ++i) {
    if (_Edges->at(i) == edge) {
      _Edges->erase(_Edges->begin() + i);
      return;
    }
  }
}

}

}
