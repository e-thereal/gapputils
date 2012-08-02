#ifndef PROPERTYREFERENCE_H
#define PROPERTYREFERENCE_H

#include <QObject>
#include <qmetatype.h>

#include <capputils/ReflectableClass.h>

namespace gapputils {

namespace workflow {

class Workflow;
class Node;

}

}

/**
 * A property can be identified by its workflow, the node's UUID and the canonical property name
 */

class PropertyReference /*: public QObject*/
{
  //Q_OBJECT

private:

  // Identification properties

  const gapputils::workflow::Workflow* workflow;  ///< Workflow
  std::string nodeId;         ///< The UUID of the node associated with the property
  std::string propertyId;     ///< PropertyName of the format propName.subPropName

  // Runtime access properties

  gapputils::workflow::Node* node;
  capputils::reflection::ReflectableClass* object;
  capputils::reflection::IClassProperty* prop;

public:
  PropertyReference();
  PropertyReference(const gapputils::workflow::Workflow* workflow,
      const std::string& nodeId,
      const std::string& propertyId);
  virtual ~PropertyReference();

public:
  const gapputils::workflow::Workflow* getWorkflow() const;
  std::string getNodeId() const;
  std::string getPropertyId() const;

  gapputils::workflow::Node* getNode() const;
  capputils::reflection::ReflectableClass* getObject() const;
  capputils::reflection::IClassProperty* getProperty() const;
};

Q_DECLARE_METATYPE(PropertyReference)

//class ConstPropertyReference /*: public QObject*/
//{
//  //Q_OBJECT
//
//private:
//  const capputils::reflection::ReflectableClass* object;
//  const capputils::reflection::IClassProperty* prop;
//  const gapputils::workflow::Node* node;
//
//public:
//  ConstPropertyReference();
//  ConstPropertyReference(const capputils::reflection::ReflectableClass* object,
//      const capputils::reflection::IClassProperty* prop,
//      const gapputils::workflow::Node* node);
//  virtual ~ConstPropertyReference();
//
//public:
//  const capputils::reflection::ReflectableClass* getObject() const;
//  const capputils::reflection::IClassProperty* getProperty() const;
//  const gapputils::workflow::Node* getNode() const;
//  void setNode(const gapputils::workflow::Node* node);
//};

//Q_DECLARE_METATYPE(ConstPropertyReference)

#endif // PROPERTYREFERENCE_H
