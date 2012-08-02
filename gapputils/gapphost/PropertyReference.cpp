#include "PropertyReference.h"

#include "Node.h"
#include "Workflow.h"
#include <gapputils/CollectionElement.h>
#include <capputils/IReflectableAttribute.h>

#include <cassert>

using namespace capputils::reflection;

PropertyReference::PropertyReference() : workflow(0), node(0), object(0), prop(0) { }

PropertyReference::PropertyReference(const gapputils::workflow::Workflow* workflow,
    const std::string& nodeId,
    const std::string& propertyId)
 : workflow(workflow), nodeId(nodeId), propertyId(propertyId), node(0), object(0), prop(0)
{
  using namespace gapputils::workflow;

  node = workflow->getNode(nodeId);
  assert(node);

  Workflow* subworkflow = dynamic_cast<Workflow*>(node);

  std::string propertyPath = propertyId;
  size_t pos = propertyPath.find_first_of('.');
  std::string propertyName = propertyPath.substr(0, pos);
  if (pos != std::string::npos)
    propertyPath = propertyPath.substr(pos + 1);
  else
    propertyPath.clear();

  object = node->getModule();
  assert(object);
  prop = object->findProperty(propertyName);

  if (!prop && subworkflow) {
    std::vector<Node*>& interfaceNodes = subworkflow->getInterfaceNodes();
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (interfaceNodes[i]->getUuid() == propertyName) {
        object = interfaceNodes[i]->getModule();
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

PropertyReference::~PropertyReference()
{
}

const gapputils::workflow::Workflow* PropertyReference::getWorkflow() const {
  return workflow;
}

std::string PropertyReference::getNodeId() const {
  return nodeId;
}

std::string PropertyReference::getPropertyId() const {
  return propertyId;
}

gapputils::workflow::Node* PropertyReference::getNode() const {
  return node;
}

capputils::reflection::ReflectableClass* PropertyReference::getObject() const {
  return object;
}

capputils::reflection::IClassProperty* PropertyReference::getProperty() const {
  return prop;
}
