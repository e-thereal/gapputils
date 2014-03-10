#include "PropertyReference.h"

#include "Node.h"
#include "Workflow.h"
#include <gapputils/CollectionElement.h>
#include <capputils/attributes/IReflectableAttribute.h>

#include <cassert>

using namespace capputils::reflection;

PropertyReference::PropertyReference() : object(0), prop(0) { }

PropertyReference::PropertyReference(boost::shared_ptr<const gapputils::workflow::Workflow> workflow,
    const std::string& nodeId,
    const std::string& propertyId)
 : workflow(workflow), nodeId(nodeId), propertyId(propertyId), object(0), prop(0)
{
  using namespace gapputils::workflow;

  node = workflow->getNode(nodeId);
  assert(!node.expired());

  boost::shared_ptr<Workflow> subworkflow = boost::dynamic_pointer_cast<Workflow>(node.lock());

  std::string propertyPath = propertyId;
  size_t pos = propertyPath.find_first_of('.');
  std::string propertyName = propertyPath.substr(0, pos);
  if (pos != std::string::npos)
    propertyPath = propertyPath.substr(pos + 1);
  else
    propertyPath.clear();

  object = node.lock()->getModule().get();
  assert(object);
  prop = object->findProperty(propertyName);

  if (!prop && subworkflow) {
    std::vector<boost::weak_ptr<Node> >& interfaceNodes = subworkflow->getInterfaceNodes();
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (interfaceNodes[i].lock()->getUuid() == propertyName) {
        object = interfaceNodes[i].lock()->getModule().get();
        assert(object);
        if (dynamic_cast<CollectionElement*>(object))
          prop = object->findProperty("Values");
        else
          prop = object->findProperty("Value");
        break;
      }
    }
  }
  assert(prop);

  capputils::attributes::IReflectableAttribute* reflectable = 0;
  while (propertyPath.size() &&
      (reflectable = prop->getAttribute<capputils::attributes::IReflectableAttribute>()) &&
      (object = reflectable->getValuePtr(*object, prop)))
  {
    pos = propertyPath.find_first_of('.');
    propertyName = propertyPath.substr(0, pos);
    if (pos != std::string::npos)
      propertyPath = propertyPath.substr(pos);
    else
      propertyPath.clear();

    prop = object->findProperty(propertyName);
    assert(prop);
  }
  assert(propertyPath.size() == 0);
}

PropertyReference::~PropertyReference() { }

boost::shared_ptr<PropertyReference> PropertyReference::TryCreate(
    boost::shared_ptr<const gapputils::workflow::Workflow> workflow,
    const std::string& nodeId,
    const std::string& propertyId)
{
  using namespace gapputils::workflow;

  boost::weak_ptr<gapputils::workflow::Node> node = workflow->getNode(nodeId);
  capputils::reflection::ReflectableClass* object = 0;
  capputils::reflection::IClassProperty* prop = 0;

  node = workflow->getNode(nodeId);
  if (node.expired())
    return boost::shared_ptr<PropertyReference>();

  boost::shared_ptr<Workflow> subworkflow = boost::dynamic_pointer_cast<Workflow>(node.lock());

  std::string propertyPath = propertyId;
  size_t pos = propertyPath.find_first_of('.');
  std::string propertyName = propertyPath.substr(0, pos);
  if (pos != std::string::npos)
    propertyPath = propertyPath.substr(pos + 1);
  else
    propertyPath.clear();

  object = node.lock()->getModule().get();
  assert(object);

  prop = object->findProperty(propertyName);
  if (!prop && subworkflow) {
    std::vector<boost::weak_ptr<Node> >& interfaceNodes = subworkflow->getInterfaceNodes();
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (interfaceNodes[i].lock()->getUuid() == propertyName) {
        object = interfaceNodes[i].lock()->getModule().get();
        assert(object);
        if (dynamic_cast<CollectionElement*>(object))
          prop = object->findProperty("Values");
        else
          prop = object->findProperty("Value");
        break;
      }
    }
  }
  if (!prop)
    return boost::shared_ptr<PropertyReference>();

  capputils::attributes::IReflectableAttribute* reflectable = 0;
  while (propertyPath.size() &&
      (reflectable = prop->getAttribute<capputils::attributes::IReflectableAttribute>()) &&
      (object = reflectable->getValuePtr(*object, prop)))
  {
    pos = propertyPath.find_first_of('.');
    propertyName = propertyPath.substr(0, pos);
    if (pos != std::string::npos)
      propertyPath = propertyPath.substr(pos);
    else
      propertyPath.clear();

    prop = object->findProperty(propertyName);
    if (!prop)
      return boost::shared_ptr<PropertyReference>();
  }
  if (propertyPath.size() != 0)
    return boost::shared_ptr<PropertyReference>();

  boost::shared_ptr<PropertyReference> ref(new PropertyReference);
  ref->workflow = workflow;
  ref->nodeId = nodeId;
  ref->propertyId = propertyId;
  ref->node = node;
  ref->object = object;
  ref->prop = prop;

  return ref;
}

boost::shared_ptr<const gapputils::workflow::Workflow> PropertyReference::getWorkflow() const {
  return workflow.lock();
}

std::string PropertyReference::getNodeId() const {
  return nodeId;
}

std::string PropertyReference::getPropertyId() const {
  return propertyId;
}

boost::shared_ptr<gapputils::workflow::Node> PropertyReference::getNode() const {
  return node.lock();
}

capputils::reflection::ReflectableClass* PropertyReference::getObject() const {
  return object;
}

capputils::reflection::IClassProperty* PropertyReference::getProperty() const {
  return prop;
}
