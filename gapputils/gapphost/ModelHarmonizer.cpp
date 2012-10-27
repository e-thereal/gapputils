/*
 * ModelHarmonizer.cpp
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#include "ModelHarmonizer.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/AbstractEnumerator.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <iostream>
#include <sstream>
#include <gapputils/LabelAttribute.h>
#include <gapputils/HideAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include "Workflow.h"
#include <gapputils/WorkflowElement.h>

#include "Node.h"

#include <cassert>

#include "PropertyReference.h"

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;
using namespace std;

namespace gapputils {

using namespace attributes;
using namespace workflow;

namespace host {

void buildModel(QStandardItem* parentItem, ReflectableClass* object, Node* node, const std::string& propertyPrefix = "");

void addPropertyRow(PropertyReference& ref, QStandardItem* parentItem, int gridPos, const std::string& name = "")
{
  IClassProperty* property = ref.getProperty();
  boost::shared_ptr<Node> node = ref.getNode();
  string keyName = name;
  if (!keyName.size()) {
    keyName = property->getName();
    ShortNameAttribute* shortName = property->getAttribute<ShortNameAttribute>();
    if (shortName)
      keyName = keyName + " (" + shortName->getName() + ")";
  }
  QStandardItem* key = new QStandardItem(keyName.c_str());
  QStandardItem* value = new QStandardItem();
  key->setEditable(false);
  value->setData(QVariant::fromValue(ref), Qt::UserRole);

  if (node->getGlobalProperty(ref)) {
    QFont font = value->font();
    font.setUnderline(true);
    value->setFont(font);
  }

  if (node->getGlobalEdge(ref)) {
    QFont font = value->font();
    font.setItalic(true);
    value->setFont(font);
  }

  DescriptionAttribute* description = property->getAttribute<DescriptionAttribute>();
  if (description) {
    key->setToolTip(description->getDescription().c_str());
    value->setToolTip(description->getDescription().c_str());
  }

  if (property->getAttribute<LabelAttribute>()) {
    QFont font = key->font();
    font.setBold(true);
    key->setFont(font);
  }

  if (property->getAttribute<ReadOnlyAttribute>()) {
    value->setEditable(false);
  }

  IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
  if (reflectable) {
    ReflectableClass* subObject = reflectable->getValuePtr(*ref.getObject(), property);

    // If the type of the reflectable object has changed, the subtree needs to be rebuild.
    // You need to know the previous type in order to detect a changed. ScalarAttributes are
    // no longer supported in order to guarantee, that the string value is always set to the
    // previous type name.

    if (subObject) {
      value->setText(subObject->getClassName().c_str());
      value->setEnabled(false);
      buildModel(key, subObject, node.get(), ref.getPropertyId() + ".");
    } else {
      // TODO: report problem
    }
  } else {
    value->setText(property->getStringValue(*ref.getObject()).c_str());
  }
  parentItem->setChild(gridPos, 0, key);
  parentItem->setChild(gridPos, 1, value);
}

void buildModel(QStandardItem* parentItem, ReflectableClass* object, Node* node, const std::string& propertyPrefix) {
  vector<IClassProperty*> properties = object->getProperties();
  parentItem->removeRows(0, parentItem->rowCount());

  unsigned gridPos = 0;
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<HideAttribute>())
      continue;

    PropertyReference ref(node->getWorkflow().lock(), node->getUuid(), propertyPrefix + properties[i]->getName());
    addPropertyRow(ref, parentItem, gridPos);
    ++gridPos;
  }

  // If top level and workflow, add interface properties
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow && node->getModule().get() == object) {
    std::vector<boost::weak_ptr<Node> >& nodes = workflow->getInterfaceNodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
      PropertyReference ref(node->getWorkflow().lock(), node->getUuid(), propertyPrefix + nodes[i].lock()->getUuid());
      WorkflowElement* element = dynamic_cast<WorkflowElement*>(nodes[i].lock()->getModule().get());
      if (element)
        addPropertyRow(ref, parentItem, gridPos, element->getLabel());
      else
        addPropertyRow(ref, parentItem, gridPos);
      ++gridPos;
    }
  }
}

/**
 * TODO: Implement the update of interface properties.
 */

void updateModel(QStandardItem* parentItem, ReflectableClass& object, Node* node, const std::string& propertyPrefix = "") {
  vector<IClassProperty*> properties = object.getProperties();

  for (unsigned i = 0, gridPos = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<HideAttribute>())
      continue;

    QStandardItem* value = parentItem->child(gridPos, 1);
    if (!value) {
      string keyName = properties[i]->getName();
      ShortNameAttribute* shortName = properties[i]->getAttribute<ShortNameAttribute>();
      if (shortName)
        keyName = keyName + " (" + shortName->getName() + ")";
      QStandardItem* key = new QStandardItem(keyName.c_str());
      value = new QStandardItem();
      parentItem->setChild(gridPos, 0, key);
      parentItem->setChild(gridPos, 1, value);
    }

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = reflectable->getValuePtr(object, properties[i]);

      // If the type of the reflectable object has changed, the subtree needs to be rebuild.
      // You need to know the previous type in order to detect a changed. ScalarAttributes are
      // no longer supported in order to guarantee, that the string value is always set to the
      // previous type name.
      if (subObject) {
        std::string oldClassName(value->text().toAscii().data());
        if (oldClassName.compare(subObject->getClassName())) {
          value->setText(subObject->getClassName().c_str());
          buildModel(parentItem->child(gridPos, 0), subObject, node, propertyPrefix + properties[i]->getName() + ".");
        } else {
          updateModel(parentItem->child(gridPos, 0), *subObject, node, propertyPrefix + properties[i]->getName() + ".");
        }
      } else {
        value->setText(properties[i]->getStringValue(object).c_str());
      }
    } else {
      value->setText(properties[i]->getStringValue(object).c_str());
    }
    ++gridPos;
  }
}

ModelHarmonizer::ModelHarmonizer(boost::shared_ptr<gapputils::workflow::Node> node)
 : QObject(), node(node), modelLocked(false), handler(this, &ModelHarmonizer::changedHandler)
{
  model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));

  assert(node);
  assert(node->getModule());

  buildModel(model->invisibleRootItem(), node->getModule().get(), node.get());
  connect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(itemChanged(QStandardItem*)));
  ObservableClass* observable = dynamic_cast<ObservableClass*>(node->getModule().get());
  if (observable) {
    observable->Changed.connect(handler);
  }
}

ModelHarmonizer::~ModelHarmonizer() {
  disconnect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(itemChanged(QStandardItem*)));
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  if (node) {
    ObservableClass* observable = dynamic_cast<ObservableClass*>(node->getModule().get());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
  }
  delete model;
}

QStandardItemModel* ModelHarmonizer::getModel() const {
  return model;
}

void ModelHarmonizer::changedHandler(capputils::ObservableClass* sender, int eventId) {
  modelLocked = true;
  updateModel(model->invisibleRootItem(), *node.lock()->getModule(), node.lock().get());
  modelLocked = false;
}

void ModelHarmonizer::itemChanged(QStandardItem* item) {
  if (modelLocked)
    return;

  // Update model if necessary
  if (item->data(Qt::UserRole).canConvert<PropertyReference>()) {
    const PropertyReference& reference = item->data(Qt::UserRole).value<PropertyReference>();
    ReflectableClass* object = reference.getObject();
    IClassProperty* prop = reference.getProperty();
    QString qstr = item->text();
    std::string str(qstr.toUtf8().data());
    if (prop->getStringValue(*object).compare(str)) {
      IReflectableAttribute* reflectable = prop->getAttribute<IReflectableAttribute>();
      if (reflectable) {
        ReflectableClass* subObject = reflectable->getValuePtr(*object, prop);
        if (dynamic_cast<AbstractEnumerator*>(subObject)) {
          stringstream stream(str);
          subObject->fromStream(stream);
          reflectable->setValuePtr(*object, prop, subObject);
        }
      } else {
        prop->setStringValue(*object, str);
      }
    }
  }
}

}

}
