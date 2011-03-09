#include "PropertyReference.h"

using namespace capputils::reflection;

PropertyReference::PropertyReference()
{
}

PropertyReference::PropertyReference(ReflectableClass* object,
    IClassProperty* prop)
    : object(object), prop(prop)
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
