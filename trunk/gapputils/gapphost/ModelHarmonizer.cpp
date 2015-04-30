/*
 * ModelHarmonizer.cpp
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#include "ModelHarmonizer.h"

#include <capputils/AbstractEnumerator.h>
#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/IReflectableAttribute.h>
#include <capputils/attributes/FlagAttribute.h>
#include <capputils/attributes/ScalarAttribute.h>
#include <capputils/attributes/HideAttribute.h>
#include <gapputils/attributes/LabelAttribute.h>
#include <capputils/attributes/ShortNameAttribute.h>

#include <gapputils/WorkflowElement.h>
#include <gapputils/attributes/ReadOnlyAttribute.h>
#include <gapputils/attributes/GroupAttribute.h>

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <map>

#include "Workflow.h"
#include "Node.h"
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

void addPropertyRow(PropertyReference& ref, QStandardItem* parentItem, const std::string& name = "")
{
  size_t gridPos = parentItem->rowCount();

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
  key->setData(QVariant::fromValue(gridPos), Qt::UserRole);
  value->setData(QVariant::fromValue(ref), Qt::UserRole);

  if (name.size()) {
    key->setFlags(key->flags() | Qt::ItemIsDragEnabled);
  } else {
    key->setFlags(key->flags() & ~Qt::ItemIsDragEnabled);
  }

  key->setFlags(key->flags() & ~Qt::ItemIsDropEnabled);
  value->setFlags(value->flags() & ~Qt::ItemIsDropEnabled);
  value->setFlags(value->flags() & ~Qt::ItemIsDragEnabled);

  if (node->getWorkflow().lock()->getGlobalProperty(ref)) {
    QFont font = value->font();
    font.setUnderline(true);
    value->setFont(font);
  }

  if (node->getWorkflow().lock()->getGlobalEdge(ref)) {
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
//    font.setBold(true);
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
  } else if (property->getAttribute<FlagAttribute>()) {
    value->setEditable(false);
    value->setCheckable(true);
    ClassProperty<bool>* boolProperty = dynamic_cast<ClassProperty<bool>* >(property);

    // properties tagged as Flag() must be of type bool
    assert(boolProperty);
    if (boolProperty->getValue(*ref.getObject()))
      value->setCheckState(Qt::Checked);
    else
      value->setCheckState(Qt::Unchecked);
  } else {
    value->setText(property->getStringValue(*ref.getObject()).c_str());
  }
  parentItem->setChild(gridPos, 0, key);
  parentItem->setChild(gridPos, 1, value);
}

QStandardItem* addGroupItem(const std::string& name, QStandardItem* parentItem) {
  size_t gridPos = parentItem->rowCount();

  // Add group label
  QStandardItem* groupItem = new QStandardItem(name.c_str());
  groupItem->setEditable(false);
  groupItem->setFlags(groupItem->flags() & ~Qt::ItemIsDropEnabled);
  groupItem->setFlags(groupItem->flags() & ~Qt::ItemIsDragEnabled);
  groupItem->setSelectable(false);

  QLinearGradient gradient(0, 0, 250, 0);
  gradient.setColorAt(1, Qt::white);
  gradient.setColorAt(0, QColor(220, 220, 255));
//  gradient.setColorAt(0, QColor(220, 220, 220));
//  groupItem->setBackground(gradient);

  QFont font = groupItem->font();
  font.setBold(true);
  groupItem->setFont(font);
  groupItem->setSizeHint(QSize(300, 20));
//  groupItem->setTextAlignment(Qt::AlignHCenter);
  parentItem->setChild(gridPos, 0, groupItem);

  QStandardItem* groupValue = new QStandardItem();
  groupValue->setEditable(false);
  groupValue->setFlags(groupItem->flags() & ~Qt::ItemIsDropEnabled);
  groupValue->setFlags(groupItem->flags() & ~Qt::ItemIsDragEnabled);
  groupValue->setSelectable(false);
//  groupValue->setBackground(Qt::white);
//  parentItem->setChild(gridPos, 1, groupValue);

  return groupItem;
}

void buildModel(QStandardItem* parentItem, ReflectableClass* object, Node* node, const std::string& propertyPrefix) {
  vector<IClassProperty*> properties = object->getProperties();
  parentItem->removeRows(0, parentItem->rowCount());


  if (!propertyPrefix.size()) {
    std::map<std::string, std::vector<PropertyReference> > references;
    std::vector<std::string> groups;

    // Collect groups and references per group
    for (size_t iProp = 0; iProp < properties.size(); ++iProp) {
      IClassProperty* prop = properties[iProp];

      if (prop->getAttribute<HideAttribute>())
        continue;

      GroupAttribute* groupAttr = prop->getAttribute<GroupAttribute>();
      std::string group = (groupAttr ? groupAttr->getName() : "General");

      if (references.find(group) == references.end()) {
        groups.push_back(group);
        references[group] = std::vector<PropertyReference>();
      }

      PropertyReference ref(node->getWorkflow().lock(), node->getUuid(), propertyPrefix + prop->getName());
      references[group].push_back(ref);
    }

    // Create items for each references and group
    for (size_t iGroup = 0; iGroup < groups.size(); ++iGroup) {
      const std::string group = groups[iGroup];
      std::vector<PropertyReference>& refs = references[group];
      addGroupItem(group, parentItem);
      for (size_t iRef = 0; iRef < refs.size(); ++iRef) {
        addPropertyRow(refs[iRef], parentItem);
      }
    }
  } else {
    for (unsigned i = 0; i < properties.size(); ++i) {
      if (properties[i]->getAttribute<HideAttribute>())
        continue;

      PropertyReference ref(node->getWorkflow().lock(), node->getUuid(), propertyPrefix + properties[i]->getName());
  //    addPropertyRow(ref, groupItems["General"], propertyPrefix);
      addPropertyRow(ref, parentItem);
    }
  }


  // If top level and workflow, add interface properties
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow && node->getModule().get() == object) {

    std::vector<boost::weak_ptr<Node> >& nodes = workflow->getInterfaceNodes();
    if (nodes.size()) {
      addGroupItem("Interface properties", parentItem);
//      parentItem = addGroupItem("Interface properties", parentItem);
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
      PropertyReference ref(node->getWorkflow().lock(), node->getUuid(), propertyPrefix + nodes[i].lock()->getUuid());
      WorkflowElement* element = dynamic_cast<WorkflowElement*>(nodes[i].lock()->getModule().get());
      if (element)
        addPropertyRow(ref, parentItem, element->getLabel());
      else
        addPropertyRow(ref, parentItem);
    }
  }
}

void updateModel(QStandardItem* parentItem, const std::string& propertyPrefix = "") {
  // Go throw keys of parentItem

  for (int iRow = 0; iRow < parentItem->rowCount(); ++iRow) {
    QStandardItem* valueItem = parentItem->child(iRow, 1);
    if (valueItem && valueItem->data(Qt::UserRole).canConvert<PropertyReference>()) {
      PropertyReference ref = valueItem->data(Qt::UserRole).value<PropertyReference>();

      IReflectableAttribute* reflectable = ref.getProperty()->getAttribute<IReflectableAttribute>();
      if (reflectable) {
        ReflectableClass* subObject = reflectable->getValuePtr(*ref.getObject(), ref.getProperty());

        // If the type of the reflectable object has changed, the subtree needs to be rebuild.
        // You need to know the previous type in order to detect a change. ScalarAttributes are
        // no longer supported in order to guarantee, that the string value is always set to the
        // previous type name.
        if (subObject) {
          std::string oldClassName(valueItem->text().toAscii().data());
          if (oldClassName.compare(subObject->getClassName())) {
            valueItem->setText(subObject->getClassName().c_str());
            buildModel(parentItem->child(iRow, 0), subObject, ref.getNode().get(), propertyPrefix + ref.getProperty()->getName() + ".");
          } else {
            updateModel(parentItem->child(iRow, 0), propertyPrefix + ref.getProperty()->getName() + ".");
          }
        } else {
          valueItem->setText(ref.getProperty()->getStringValue(*ref.getObject()).c_str());
        }
      } else if (ref.getProperty()->getAttribute<FlagAttribute>()) {
        ClassProperty<bool>* boolProperty = dynamic_cast<ClassProperty<bool>* >(ref.getProperty());
        // properties tagged as Flag() must be of type bool
        assert(boolProperty);
        if (boolProperty->getValue(*ref.getObject()))
          valueItem->setCheckState(Qt::Checked);
        else
          valueItem->setCheckState(Qt::Unchecked);
      } else{
        valueItem->setText(ref.getProperty()->getStringValue(*ref.getObject()).c_str());
      }
    } else {
      if (parentItem->child(iRow, 0)->hasChildren()) {
        updateModel(parentItem->child(iRow, 0), propertyPrefix);
      }
    }
  }
}

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
      // You need to know the previous type in order to detect a change. ScalarAttributes are
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
    } else if (properties[i]->getAttribute<FlagAttribute>()) {
      ClassProperty<bool>* boolProperty = dynamic_cast<ClassProperty<bool>* >(properties[i]);
      // properties tagged as Flag() must be of type bool
      assert(boolProperty);
      if (boolProperty->getValue(object))
        value->setCheckState(Qt::Checked);
      else
        value->setCheckState(Qt::Unchecked);
    } else{
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

void ModelHarmonizer::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {
  modelLocked = true;
//  updateModel(model->invisibleRootItem(), *node.lock()->getModule(), node.lock().get());
  updateModel(model->invisibleRootItem());
  modelLocked = false;
}

void ModelHarmonizer::itemChanged(QStandardItem* item) {
  if (!item)
    return;

  if (modelLocked)
    return;

  if (item->data(Qt::UserRole).canConvert<int>()) {
    int from = item->data(Qt::UserRole).value<int>();
    int to = item->index().row();

    // Delete new row if new row (new row if value does not have a property reference)
    if (!model->item(to, 1) || !model->item(to, 1)->data(Qt::UserRole).canConvert<PropertyReference>()) {
      model->removeRow(to, item->index().parent());

      if (from != to) {

        // move rows to first valid position (count number of workflow module properties)
        // Update grid positions of all keys
        // Update workflow interface node order

        boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();

        int firstInterfaceProperty = 0;

        for (int iRow = 0; iRow < model->rowCount(); ++iRow) {
          if (model->item(iRow, 1) && model->item(iRow, 1)->data(Qt::UserRole).canConvert<PropertyReference>()) {
            PropertyReference ref = model->item(iRow, 1)->data(Qt::UserRole).value<PropertyReference>();
            if (ref.getObject() != node->getModule().get()) {
              firstInterfaceProperty = iRow;
              break;
            }
          }
        }

        to = std::max(to, firstInterfaceProperty);
        if (from < to)
          --to;
        model->insertRow(to, model->takeRow(from));

        modelLocked = true;
        for (int gridPos = 0; gridPos < model->rowCount(); ++gridPos)
          model->item(gridPos, 0)->setData(QVariant::fromValue(gridPos), Qt::UserRole);
        modelLocked = false;

        gapputils::Workflow* workflow = dynamic_cast<gapputils::Workflow*>(node.get());
        if (workflow) {
          workflow->moveInterfaceNode(from - firstInterfaceProperty, to - firstInterfaceProperty);
        }
      }
      return;
    }
  }

  // Update model if necessary
  if (item->data(Qt::UserRole).canConvert<PropertyReference>()) {
    const PropertyReference& reference = item->data(Qt::UserRole).value<PropertyReference>();
    ReflectableClass* object = reference.getObject();
    IClassProperty* prop = reference.getProperty();
    QString qstr = item->text();
    std::string str(qstr.toUtf8().data());
    if (prop->getAttribute<FlagAttribute>()) {
      if ((prop->getStringValue(*object) != "0") != (item->checkState() == Qt::Checked)) {
        prop->setStringValue(*object, (item->checkState() == Qt::Checked ? "1" : "0"));
      }
    } else if (prop->getStringValue(*object).compare(str)) {
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
