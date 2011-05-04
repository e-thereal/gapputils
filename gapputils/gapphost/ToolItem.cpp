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
#include <ObserveAttribute.h>
#include <EventHandler.h>
#include <qapplication.h>
#include <QFontMetrics>
#include "Workbench.h"

#include "LabelAttribute.h"
#include "InputAttribute.h"
#include "OutputAttribute.h"
#include "CableItem.h"

using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

ToolConnection::ToolConnection(const QString& label, Direction direction, ToolItem* parent, IClassProperty* property)
  : x(0), y(0), width(6), height(7), label(label), direction(direction), parent(parent), cable(0), property(property)
{
  ObserveAttribute* observe = property->getAttribute<ObserveAttribute>();
  if (observe) {
    propertyId = observe->getEventId();
  } else {
    propertyId = -1;
  }
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

void ToolConnection::draw(QPainter* painter, bool showLabel) const {
  CableItem* currentCable = parent->bench->getCurrentCable();
  painter->save();
  painter->setPen(Qt::black);
  switch (direction) {
  case Input:
    if (currentCable && currentCable->needOutput() && currentCable->getInput()->property->getType() == property->getType()) {
      painter->setBrush(Qt::white);
    } else if(parent->isSelected() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getOutput() == this) {
      painter->setBrush(Qt::white);
    } else if (!currentCable && cable && cable->getInput()->parent->isSelected()) {
      painter->setBrush(Qt::white);
    } else {
      painter->setBrush(Qt::darkGray);
    }
    painter->drawRect(x - width, y - (height/2)-1, width, height + 1);
    if (showLabel)
      painter->drawText(x + 6, y - 10, 100, 20, Qt::AlignVCenter, label);
    break;

  case Output:
    if (currentCable && currentCable->needInput() && currentCable->getOutput()->property->getType() == property->getType()) {
      painter->setBrush(Qt::white);
    } else if(parent->isSelected() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getInput() == this) {
      painter->setBrush(Qt::white);
    } else if (!currentCable && cable && cable->getOutput()->parent->isSelected()) {
      painter->setBrush(Qt::white);
    } else {
      painter->setBrush(Qt::darkGray);
    }
    painter->drawRect(x, y - (height/2)-1, width, height + 1);
    if (showLabel)
      painter->drawText(x - 6 - 100, y - 10, 100, 20, Qt::AlignVCenter | Qt::AlignRight, label);
    break;
  }
  painter->restore();
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

ToolItem::ToolItem(workflow::Node* node, Workbench *bench)
 : node(node), bench(bench), harmonizer(node->getModule()),
   width(190), height(90), adjust(3 + 10), connectionDistance(16), inputsWidth(0),
   labelWidth(35), outputsWidth(0), labelFont(QApplication::font())
{
  setFlag(ItemIsMovable);
  // TODO: check if this causes problems
  //setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  labelFont.setBold(true);
  labelFont.setPointSize(10);

  ReflectableClass* object = node->getModule();
  ObservableClass* observable = dynamic_cast<ObservableClass*>(object);
  if (observable)
    observable->Changed.connect(capputils::EventHandler<ToolItem>(this, &ToolItem::changedHandler));

  updateConnections();
  updateSize();
}

ToolItem::~ToolItem() {
  for (unsigned i = 0; i < inputs.size(); ++i)
    delete inputs[i];
  for (unsigned i = 0; i < outputs.size(); ++i)
    delete outputs[i];
}

void ToolItem::updateConnections() {
  for (unsigned i = 0; i < inputs.size(); ++i)
    delete inputs[i];
  for (unsigned i = 0; i < outputs.size(); ++i)
    delete outputs[i];
  inputs.clear();
  outputs.clear();

  ReflectableClass* object = node->getModule();
  vector<IClassProperty*>& properties = object->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<InputAttribute>()) {
      inputs.push_back(new ToolConnection(properties[i]->getName().c_str(), ToolConnection::Input, this, properties[i]));
    }
    if (properties[i]->getAttribute<OutputAttribute>()) {
      outputs.push_back(new ToolConnection(properties[i]->getName().c_str(), ToolConnection::Output, this, properties[i]));
    }
  }
}

void ToolItem::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {
  updateSize();
  update();
}

void ToolItem::setWorkbench(Workbench* bench) {
  this->bench = bench;
}

Node* ToolItem::getNode() const {
  return node;
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

ToolConnection* ToolItem::getConnection(const std::string& propertyName, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i]->property->getName().compare(propertyName) == 0)
        return inputs[i];
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i]->property->getName().compare(propertyName) == 0)
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
    if (bench)
      bench->notifyItemChange(this);
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

void ToolItem::updateSize() {
  QFontMetrics fontMetrics(QApplication::font());
  QFontMetrics labelFontMetrics(labelFont);
  inputsWidth = outputsWidth = 0;
  for (unsigned i = 0; i < inputs.size(); ++i)
    inputsWidth = max(inputsWidth, fontMetrics.boundingRect(inputs[i]->label).width() + 8);
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputsWidth = max(outputsWidth, fontMetrics.boundingRect(outputs[i]->label).width() + 8);

  width = (inputs.size() ? inputsWidth : 0) + labelWidth + (outputs.size() ? outputsWidth : 0);
  height = 30;
  height = max(height, labelFontMetrics.boundingRect(getLabel().c_str()).width() + 20);
  height = max(height, connectionDistance * (int)inputs.size() + 12);
  height = max(height, connectionDistance * (int)outputs.size() + 12);
  updateConnectionPositions();
}

void ToolItem::updateConnectionPositions() {
  for (int i = 0, pos = -((int)inputs.size()-1) * connectionDistance / 2 + height/2; i < (int)inputs.size(); ++i, pos += connectionDistance) {
    inputs[i]->setPos(0, pos);
  }
  for (int i = 0, pos = -((int)outputs.size()-1) * connectionDistance / 2 + height/2; i < (int)outputs.size(); ++i, pos += connectionDistance) {
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

void ToolItem::drawConnections(QPainter* painter, bool showLabel) {
  for (unsigned i = 0; i < inputs.size(); ++i)
    inputs[i]->draw(painter, showLabel);
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputs[i]->draw(painter, showLabel);
}

void ToolItem::drawBox(QPainter* painter) {
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
}

std::string ToolItem::getLabel() const {
  vector<IClassProperty*>& properties = getNode()->getModule()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      return properties[i]->getStringValue(*getNode()->getModule());
    }
  }
  return string("[") + getNode()->getModule()->getClassName() + "]";
}

void ToolItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
 {
  Q_UNUSED(option)

  QString label = getLabel().c_str();

  drawBox(painter);
  drawConnections(painter);
 
  painter->save();
  painter->translate((inputs.size() ? inputsWidth : 0) + labelWidth/2, height/2);
  //painter->translate(width/2, height/2);
  painter->rotate(270);
  painter->translate(-(inputs.size() ? inputsWidth : 0) -labelWidth/2, -height/2);
  //painter->translate(-width/2, -height/2);
  painter->setFont(labelFont);
  painter->drawText((inputs.size() ? inputsWidth : 0) - height, 0, labelWidth + 2 * height, height, Qt::AlignCenter, label);
  //painter->drawText(0, 0, width, height, Qt::AlignCenter, label);
  painter->restore();
 }

}
