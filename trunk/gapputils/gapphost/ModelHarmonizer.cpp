/*
 * ModelHarmonizer.cpp
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#include "ModelHarmonizer.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/Enumerator.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <iostream>
#include <sstream>
#include <gapputils/LabelAttribute.h>
#include <gapputils/HideAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

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

void buildModel(QStandardItem* parentItem, ReflectableClass* object, Node* node) {
  vector<IClassProperty*> properties = object->getProperties();
  parentItem->removeRows(0, parentItem->rowCount());

  for (unsigned i = 0, gridPos = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<HideAttribute>())
      continue;

    string keyName = properties[i]->getName();
    ShortNameAttribute* shortName = properties[i]->getAttribute<ShortNameAttribute>();
    if (shortName)
      keyName = keyName + " (" + shortName->getName() + ")";
    QStandardItem* key = new QStandardItem(keyName.c_str());
    QStandardItem* value = new QStandardItem();
    key->setEditable(false);
    value->setData(QVariant::fromValue(PropertyReference(object, properties[i], node)), Qt::UserRole);

    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description) {
      key->setToolTip(description->getDescription().c_str());
      value->setToolTip(description->getDescription().c_str());
    }

    if (properties[i]->getAttribute<LabelAttribute>()) {
      QFont font = key->font();
      font.setBold(true);
      key->setFont(font);
    }

    if (properties[i]->getAttribute<ReadOnlyAttribute>()) {
      value->setEditable(false);
    }

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = reflectable->getValuePtr(*object, properties[i]);

      Enumerator* enumerator = dynamic_cast<Enumerator*>(subObject);
      if (!enumerator && subObject) {
        if (!subObject->getAttribute<ScalarAttribute>()) {
          value->setText(subObject->getClassName().c_str());
          value->setEnabled(false);
        } else {
          value->setText(properties[i]->getStringValue(*object).c_str());
        }
        buildModel(key, subObject, node);
      } else {
        value->setText(properties[i]->getStringValue(*object).c_str());
      }
    } else {
      value->setText(properties[i]->getStringValue(*object).c_str());
    }
    parentItem->setChild(gridPos, 0, key);
    parentItem->setChild(gridPos, 1, value);
    ++gridPos;
  }
}

void updateModel(QStandardItem* parentItem, ReflectableClass& object) {
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

      Enumerator* enumerator = dynamic_cast<Enumerator*>(subObject);
      if (!enumerator && subObject) {
        if (!subObject->getAttribute<ScalarAttribute>()) {
          value->setText(subObject->getClassName().c_str());
        } else {
          value->setText(properties[i]->getStringValue(object).c_str());
        }
        updateModel(parentItem->child(gridPos, 0), *subObject);
      } else {
        value->setText(properties[i]->getStringValue(object).c_str());
      }
    } else {
      value->setText(properties[i]->getStringValue(object).c_str());
    }
    ++gridPos;
  }
}

void ModelHarmonizer::ObjectChangedHandler::operator()(capputils::ObservableClass* /*sender*/, int /*eventId*/) {
  updateModel(parent->model->invisibleRootItem(), *parent->node->getModule());
}

ModelHarmonizer::ModelHarmonizer(gapputils::workflow::Node* node)
 : QObject(), objectChanged(this), node(node)
{
  model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));

  assert(node);
  assert(node->getModule());

  buildModel(model->invisibleRootItem(), node->getModule(), node);
  connect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(itemChanged(QStandardItem*)));
  ObservableClass* observable = dynamic_cast<ObservableClass*>(node->getModule());
  if (observable) {
    observable->Changed.connect(objectChanged);
  }
}

ModelHarmonizer::~ModelHarmonizer() {
  delete model;
}

QStandardItemModel* ModelHarmonizer::getModel() const {
  return model;
}

void ModelHarmonizer::itemChanged(QStandardItem* item) {
  // Update model if necessary
  if (item->data(Qt::UserRole).canConvert<PropertyReference>()) {
    const PropertyReference& reference = item->data(Qt::UserRole).value<PropertyReference>();
    ReflectableClass* object = reference.getNode()->getModule();
    IClassProperty* prop = reference.getProperty();
    QString qstr = item->text();
    std::string str(qstr.toUtf8().data());
    if (prop->getStringValue(*object).compare(str)) {
      IReflectableAttribute* reflectable = prop->getAttribute<IReflectableAttribute>();
      if (reflectable) {
        ReflectableClass* subObject = reflectable->getValuePtr(*object, prop);
        if (subObject->getAttribute<ScalarAttribute>()) {
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
