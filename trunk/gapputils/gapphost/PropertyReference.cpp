#include "PropertyReference.h"

using namespace capputils::reflection;

PropertyReference::PropertyReference()
{
}

PropertyReference::PropertyReference(ReflectableClass* object,
    IClassProperty* prop,
    gapputils::workflow::Node* node)
    : object(object), prop(prop), node(node)
{
}

PropertyReference::~PropertyReference()
{
}

ReflectableClass* PropertyReference::getObject() const {
  return object;
}

IClassProperty* PropertyReference::getProperty() const {
  return prop;
}

gapputils::workflow::Node* PropertyReference::getNode() const {
  return node;}
void PropertyReference::setNode(gapputils::workflow::Node* node) {
  this->node = node;
}

