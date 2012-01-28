#ifndef PROPERTYREFERENCE_H
#define PROPERTYREFERENCE_H

#include <QObject>
#include <qmetatype.h>

#include <capputils/ReflectableClass.h>

#include "Node.h"

class PropertyReference /*: public QObject*/
{
  //Q_OBJECT

private:
  capputils::reflection::ReflectableClass* object;
  capputils::reflection::IClassProperty* prop;
  gapputils::workflow::Node* node;

public:
  PropertyReference();
  PropertyReference(capputils::reflection::ReflectableClass* object,
      capputils::reflection::IClassProperty* prop,
      gapputils::workflow::Node* node);
  virtual ~PropertyReference();

public:
  capputils::reflection::ReflectableClass* getObject() const;
  capputils::reflection::IClassProperty* getProperty() const;
  gapputils::workflow::Node* getNode() const;
};

Q_DECLARE_METATYPE(PropertyReference)

#endif // PROPERTYREFERENCE_H
