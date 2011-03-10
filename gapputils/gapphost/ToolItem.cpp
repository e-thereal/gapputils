/*
 * ToolItem.cpp
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#include "ToolItem.h"

#include <QStyleOptionGraphicsItem>
#include <QPainter>
#include <QGraphicsSceneMouseEvent>
#include "Workbench.h"

#include "LabelAttribute.h"

using namespace capputils;
using namespace capputils::reflection;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

ToolItem::ToolItem(ReflectableClass* object, Workbench *bench)
 : changeHandler(this), object(object), bench(bench), harmonizer(object)
{
  setFlag(ItemIsMovable);
  setCacheMode(DeviceCoordinateCache);
  setZValue(-1);

  ObservableClass* observable = dynamic_cast<ObservableClass*>(object);
  if (observable)
    observable->Changed.connect(changeHandler);
}

ToolItem::~ToolItem() {
}

void ToolItem::setWorkbench(Workbench* bench) {
  this->bench = bench;
}

ReflectableClass* ToolItem::getObject() const {
  return object;
}

QAbstractItemModel* ToolItem::getModel() const {
  return harmonizer.getModel();
}

void ToolItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
  if (bench && bench->getSelectedItem() != this)
    bench->setSelectedItem(this);
  QGraphicsItem::mousePressEvent(event);
}

QRectF ToolItem::boundingRect() const
{
  int adjust = 3;
  return QRectF(0-adjust, 0-adjust, 90+2*adjust, 60+2*adjust);
}

void ToolItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
 {
  Q_UNUSED(option)

  QLinearGradient gradient(45, 0, 45, 60);
  if (bench && bench->getSelectedItem() == this) {
    gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(1, Qt::lightGray);
  } else {
    gradient.setColorAt(0, Qt::lightGray);
    gradient.setColorAt(1, Qt::gray);
  }

  painter->setBrush(gradient);
  painter->setPen(QPen(Qt::black, 0));
  painter->drawRoundedRect(0, 0, 90, 60, 4, 4);
  QString label;
  label.append("[");
  label.append(object->getClassName().c_str());
  label.append("]");
  vector<IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      label = properties[i]->getStringValue(*object).c_str();
      break;
    }
  }
  painter->drawText(0, 0, 90, 60, Qt::AlignCenter, label);
 }

}
