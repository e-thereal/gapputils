/*
 * ModelHarmonizer.cpp
 *
 *  Created on: Mar 9, 2011
 *      Author: tombr
 */

#include "ModelHarmonizer.h"

#include <DescriptionAttribute.h>
#include <Enumerator.h>
#include <IReflectableAttribute.h>
#include <ScalarAttribute.h>
#include <iostream>
#include <sstream>

#include "PropertyReference.h"

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;
using namespace std;

namespace gapputils {

void buildModel(QStandardItem* parentItem, ReflectableClass& object) {
  vector<IClassProperty*> properties = object.getProperties();
  parentItem->removeRows(0, parentItem->rowCount());

  for (unsigned i = 0; i < properties.size(); ++i) {
    QStandardItem *key = new QStandardItem(properties[i]->getName().c_str());
    QStandardItem* value = new QStandardItem(properties[i]->getStringValue(object).c_str());
    key->setEditable(false);
    value->setData(QVariant::fromValue(PropertyReference(&object, properties[i])), Qt::UserRole);

    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description) {
      key->setToolTip(description->getDescription().c_str());
      value->setToolTip(description->getDescription().c_str());
    }

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = reflectable->getValuePtr(object, properties[i]);

      Enumerator* enumerator = dynamic_cast<Enumerator*>(subObject);
      if (!enumerator && subObject) {
        if (!subObject->getAttribute<ScalarAttribute>()) {
          value->setText(subObject->getClassName().c_str());
          value->setEnabled(false);
        }
        buildModel(key, *subObject);
      }
    }
    parentItem->setChild(i, 0, key);
    parentItem->setChild(i, 1, value);
  }
}

void updateModel(QStandardItem* parentItem, ReflectableClass& object) {
  vector<IClassProperty*> properties = object.getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    QStandardItem* value = parentItem->child(i, 1);
    value->setText(properties[i]->getStringValue(object).c_str());

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = reflectable->getValuePtr(object, properties[i]);

      Enumerator* enumerator = dynamic_cast<Enumerator*>(subObject);
      if (!enumerator && subObject) {
        if (!subObject->getAttribute<ScalarAttribute>()) {
          value->setText(subObject->getClassName().c_str());
        }
        updateModel(parentItem->child(i, 0), *subObject);
      }
    }
  }
}

void ModelHarmonizer::ObjectChangedHandler::operator()(capputils::ObservableClass* /*sender*/, int /*eventId*/) {
  updateModel(parent->model->invisibleRootItem(), *parent->object);
}

ModelHarmonizer::ModelHarmonizer(ReflectableClass* object) : QObject(), objectChanged(this), object(object) {
  model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));

  buildModel(model->invisibleRootItem(), *object);
  connect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(itemChanged(QStandardItem*)));
  ObservableClass* observable = dynamic_cast<ObservableClass*>(object);
  if (observable) {
    observable->Changed.connect(objectChanged);
  }
}

ModelHarmonizer::~ModelHarmonizer() {
  delete model;
}

QStandardItemModel* ModelHarmonizer::getModel() {
  return model;
}

void ModelHarmonizer::itemChanged(QStandardItem* item) {
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
        stringstream stream(str);
        subObject->fromStream(stream);
        reflectable->setValuePtr(*object, prop, subObject);
      } else {
        prop->setStringValue(*object, str);
      }
    }
  }
}

}
