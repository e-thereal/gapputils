#ifndef PROPERTYREFERENCE_H
#define PROPERTYREFERENCE_H

#include <QObject>
#include <qmetatype.h>

#include <ReflectableClass.h>

class PropertyReference /*: public QObject*/
{
  //Q_OBJECT

private:
  capputils::reflection::ReflectableClass* object;
  capputils::reflection::IClassProperty* prop;

public:
  PropertyReference();
  PropertyReference(capputils::reflection::ReflectableClass* object,
      capputils::reflection::IClassProperty* prop);
  virtual ~PropertyReference();

public:
  capputils::reflection::ReflectableClass* getObject() const;
  capputils::reflection::IClassProperty* getProperty() const;
};

Q_DECLARE_METATYPE(PropertyReference)

#endif // PROPERTYREFERENCE_H
