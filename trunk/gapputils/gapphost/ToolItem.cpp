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
#include "InputAttribute.h"
#include "OutputAttribute.h"
#include "CablePlug.h"
#include "CableItem.h"

using namespace capputils;
using namespace capputils::reflection;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

ToolConnection::ToolConnection(const QString& label, Direction direction, ToolItem* parent)
  : x(0), y(0), width(6), height(7), label(label), direction(direction), parent(parent), cable(0) {
}

ToolConnection::~ToolConnection() {
  if (cable) {
    parent->bench->scene()->removeItem(cable);
    delete cable;
  }
}

void ToolConnection::connect(CableItem* cable) {
  if (this->cable) {
    CableItem* temp = this->cable;
    this->cable = 0;
    parent->bench->scene()->removeItem(temp);
    delete temp;
  }
  this->cable = cable;
}

void ToolConnection::disconnect() {
  cable = 0;
}

void ToolConnection::draw(QPainter* painter) const {
  switch (direction) {
  case Input:
    painter->setPen(Qt::black);
    painter->setBrush(Qt::darkGray);
    painter->drawRect(x - width, y - (height/2)-1, width, height + 1);
    painter->drawText(x + 6, y - 10, 100, 20, Qt::AlignVCenter, label);
    break;

  case Output:
    painter->setPen(Qt::black);
    painter->setBrush(Qt::darkGray);
    painter->drawRect(x, y - (height/2)-1, width, height + 1);
    painter->drawText(x - 6 - 100, y - 10, 100, 20, Qt::AlignVCenter | Qt::AlignRight, label);
    break;
  }
}

bool ToolConnection::hit(int x, int y) const {
  int ytol = 4;
  int xtol = 10;
  if (direction == Input)
    return this->x - width - xtol <= x && x < this->x + xtol && this->y-height/2 - ytol < y && y < this->y + height/2 + ytol;
  else
    return this->x - xtol <= x && x < this->x + width + xtol && this->y-height/2 - ytol < y && y < this->y + height/2 + ytol;
}

void ToolConnection::setPos(int x, int y) {
  this->x = x;
  this->y = y;
}

QPointF ToolConnection::attachmentPos() const {
  if (direction == Input)
    return QPointF(x-width-2, y);
  else
    return QPointF(x+width+2, y);
}

ToolItem::ToolItem(ReflectableClass* object, Workbench *bench)
 : changeHandler(this), object(object), bench(bench), harmonizer(object),
   width(130), height(60), adjust(3 + 10), connectionDistance(16)
{
  setFlag(ItemIsMovable);
  setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  ObservableClass* observable = dynamic_cast<ObservableClass*>(object);
  if (observable)
    observable->Changed.connect(changeHandler);

  vector<IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<InputAttribute>()) {
      inputs.push_back(new ToolConnection(properties[i]->getName().c_str(), ToolConnection::Input, this));
    }
    if (properties[i]->getAttribute<OutputAttribute>()) {
      outputs.push_back(new ToolConnection(properties[i]->getName().c_str(), ToolConnection::Output, this));
    }
  }
  updateConnectionPositions();
}

ToolItem::~ToolItem() {
  for (unsigned i = 0; i < inputs.size(); ++i)
    delete inputs[i];
  for (unsigned i = 0; i < outputs.size(); ++i)
    delete outputs[i];
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

ToolConnection* ToolItem::hitConnection(int x, int y, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i]->hit(x, y))
        return inputs[i];
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i]->hit(x, y))
        return outputs[i];
  }
  return 0;
}

QVariant ToolItem::itemChange(GraphicsItemChange change, const QVariant &value) {
  switch (change) {
  case ItemPositionHasChanged:
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->cable)
        inputs[i]->cable->adjust();
    }
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (outputs[i]->cable) {
        outputs[i]->cable->adjust();
      }
    }
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

void ToolItem::updateConnectionPositions() {
  for (int i = 0, pos = -((int)inputs.size()-1) * connectionDistance / 2 + height/2; i < inputs.size(); ++i, pos += connectionDistance) {
    inputs[i]->setPos(0, pos);
  }
  for (int i = 0, pos = -((int)outputs.size()-1) * connectionDistance / 2 + height/2; i < outputs.size(); ++i, pos += connectionDistance) {
    outputs[i]->setPos(width, pos);
  }
}

void ToolItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
  if (bench && bench->getSelectedItem() != this)
    bench->setSelectedItem(this);
  QGraphicsItem::mousePressEvent(event);
}

bool ToolItem::isSelected() const {
  return this == bench->getSelectedItem();
}

QRectF ToolItem::boundingRect() const
{
  return QRectF(-adjust, -adjust, width+2*adjust, height+2*adjust);
}

QPainterPath ToolItem::shape() const {
  QPainterPath path;
  path.addRect(0, 0, width, height);
  return path;
}

void ToolItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
 {
  Q_UNUSED(option)

  QLinearGradient gradient(0, 0, 0, height);
  if (bench && bench->getSelectedItem() == this) {
    gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(1, Qt::lightGray);
    setZValue(4);
  } else {
    gradient.setColorAt(0, Qt::lightGray);
    gradient.setColorAt(1, Qt::gray);
    setZValue(2);
  }

  painter->setBrush(gradient);
  painter->setPen(QPen(Qt::black, 0));
  painter->drawRoundedRect(0, 0, width, height, 4, 4);
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

  for (unsigned i = 0; i < inputs.size(); ++i)
    inputs[i]->draw(painter);
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputs[i]->draw(painter);

  painter->save();
  painter->translate(width/2, height/2);
  painter->rotate(270);
  painter->translate(-width/2, -height/2);
  QFont font = painter->font();
  font.setBold(true);
  font.setPointSize(10);
  painter->setFont(font);
  painter->drawText(0, 0, width, height, Qt::AlignCenter, label);
  painter->restore();
 }

}
