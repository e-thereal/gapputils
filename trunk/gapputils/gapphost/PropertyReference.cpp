#include "PropertyReference.h"

using namespace capputils::reflection;

PropertyReference::PropertyReference() : object(0), prop(0), node(0) { }

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

ConstPropertyReference::ConstPropertyReference() : object(0), prop(0), node(0) { }

ConstPropertyReference::ConstPropertyReference(const ReflectableClass* object,
    const IClassProperty* prop,
    const gapputils::workflow::Node* node)
    : object(object), prop(prop), node(node)
{
}

ConstPropertyReference::~ConstPropertyReference()
{
}

const ReflectableClass* ConstPropertyReference::getObject() const {
  return object;
}

const IClassProperty* ConstPropertyReference::getProperty() const {
  return prop;
}

const gapputils::workflow::Node* ConstPropertyReference::getNode() const {
  return node;}
void ConstPropertyReference::setNode(const gapputils::workflow::Node* node) {
  this->node = node;
}
