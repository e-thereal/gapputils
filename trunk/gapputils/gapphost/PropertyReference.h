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
  void setNode(gapputils::workflow::Node* node);
};

Q_DECLARE_METATYPE(PropertyReference)

class ConstPropertyReference /*: public QObject*/
{
  //Q_OBJECT

private:
  const capputils::reflection::ReflectableClass* object;
  const capputils::reflection::IClassProperty* prop;
  const gapputils::workflow::Node* node;

public:
  ConstPropertyReference();
  ConstPropertyReference(const capputils::reflection::ReflectableClass* object,
      const capputils::reflection::IClassProperty* prop,
      const gapputils::workflow::Node* node);
  virtual ~ConstPropertyReference();

public:
  const capputils::reflection::ReflectableClass* getObject() const;
  const capputils::reflection::IClassProperty* getProperty() const;
  const gapputils::workflow::Node* getNode() const;
  void setNode(const gapputils::workflow::Node* node);
};

Q_DECLARE_METATYPE(ConstPropertyReference)

#endif // PROPERTYREFERENCE_H
