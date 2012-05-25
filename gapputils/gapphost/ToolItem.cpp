/*
 * ToolItem.cpp
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

/**
 * Icons: http://icons.mysitemyway.com/category/grunge-brushed-metal-pewter-icons/
 */

#include "ToolItem.h"

#include <QStyleOptionGraphicsItem>
#include <QPainter>
#include <QGraphicsSceneMouseEvent>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <qapplication.h>
#include <QFontMetrics>
#include <qstylepainter.h>
#include "Workbench.h"

#include <gapputils/LabelAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <qpicture.h>
#include "CableItem.h"
#include <qgraphicseffect.h>

using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

using namespace workflow;

ToolConnection::ToolConnection(const QString& label, Direction direction,
    ToolItem* parent, int id, MultiConnection* multi)
  : x(0), y(0), width(6), height(7), label(label), direction(direction),
    parent(parent), id(id), cable(0), multi(multi)
{
}

ToolConnection::~ToolConnection() {
}

void ToolConnection::connect(CableItem* cable) {
  this->cable = cable;
  if (multi)
    multi->updateConnections();
}

void ToolConnection::draw(QPainter* painter, bool showLabel) const {
  // TODO: handle bundles right
  CableItem* currentCable = 0;
  vector<CableItem*>& cables = parent->bench->getCurrentCables();
  if (cables.size())
    currentCable = cables[0];

  painter->save();
  painter->setPen(Qt::black);
  QFont boldFont = painter->font();
  boldFont.setBold(true);
  switch (direction) {
  case Input:

    if (currentCable &&
        currentCable->needOutput() &&
        parent->bench->areCompatible(currentCable->getInput(), this))
    {
      painter->setBrush(Qt::white);
      painter->setOpacity(1.0);
//      painter->setFont(boldFont);
    } else if(parent->isCurrentItem() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getOutput() == this) {
      painter->setBrush(Qt::white);
    } else if (!currentCable && cable && cable->getInput()->parent->isCurrentItem()) {
      painter->setBrush(Qt::white);
    } else {
      painter->setBrush(Qt::darkGray);
    }
    painter->drawRect(x - width, y - (height/2)-1, width, height + 1);
    if (showLabel)
      painter->drawText(x + 6, y - 9, 100, 20, Qt::AlignVCenter, label);
    break;

  case Output:
    if (currentCable &&
        currentCable->needInput() &&
        parent->bench->areCompatible(this, currentCable->getOutput()))
    {
      painter->setBrush(Qt::white);
      painter->setOpacity(1.0);
//      painter->setFont(boldFont);
    } else if(parent->isCurrentItem() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getInput() == this) {
      painter->setBrush(Qt::white);
    } else if (!currentCable && cable && cable->getOutput()->parent->isCurrentItem()) {
      painter->setBrush(Qt::white);
    } else {
      painter->setBrush(Qt::darkGray);
    }
    painter->drawRect(x, y - (height/2)-1, width, height + 1);
    if (showLabel)
      painter->drawText(x - 6 - 100, y - 9, 100, 20, Qt::AlignVCenter | Qt::AlignRight, label);
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

MultiConnection::MultiConnection(const QString& label, ToolConnection::Direction direction,
    ToolItem* parent, int id)
 : label(label), direction(direction), parent(parent), id(id), expanded(false)
{
  connections.push_back(new ToolConnection(label, direction, parent, id, this));
}

MultiConnection::~MultiConnection() {
  for (unsigned i = 0; i < connections.size(); ++i)
    delete connections[i];
}

bool MultiConnection::hits(std::vector<ToolConnection*>& connections,
    int x, int y) const
{
  for (unsigned i = 0; i < this->connections.size(); ++i)
    if (this->connections[i]->hit(x, y))
      connections.push_back(this->connections[i]);
  return connections.size();
}

// TODO: deprecated. will be replaced by getConnections() or new connection
ToolConnection* MultiConnection::getLastConnection() {
  if (connections.size())
    return connections[connections.size() - 1];
  return 0;
}

void MultiConnection::adjust() {
  for (unsigned i = 0; i < connections.size(); ++i)
    if (connections[i]->cable)
      connections[i]->cable->adjust();
}

QString MultiConnection::getLabel() const {
  return label;
}

void MultiConnection::setPos(int x, int y) {
  this->x = x;
  this->y = y;
  updateConnectionPositions();
}

void MultiConnection::updateConnections() {
  for (unsigned i = 0; i < connections.size() - 1; ++i) {
    if (connections[i]->cable == 0) {
      delete connections[i];
      connections.erase(connections.begin() + i);
      --i;
    }
  }
  if (connections.size() && connections[connections.size() - 1]->cable)
    connections.push_back(new ToolConnection(label, direction, parent, id, this));
  parent->updateSize();
  parent->update();
}

void MultiConnection::updateConnectionPositions() {
  for (unsigned i = 0; i < connections.size(); ++i)
    connections[i]->setPos(x, y + (expanded ? parent->connectionDistance * i : 0));
}

void MultiConnection::draw(QPainter* painter, bool showLabel) const {
  QFontMetrics fontMetrics(QApplication::font());
  int labelWidth = fontMetrics.boundingRect(label).width();

  // TODO: if one of the connections is connected with a current cable, draw that one
  for (unsigned i = 0; i < connections.size() && (expanded ? 1 : i < 1); ++i)
    connections[i]->draw(painter, showLabel);

  QPainterPath path;
  QPolygonF polygon;
  if (expanded) {
    polygon.append(QPointF(x - labelWidth - 16, y - 3));
    polygon.append(QPointF(x - labelWidth - 10, y - 3));
    polygon.append(QPointF(x - labelWidth - 13, y + 2));
  } else {
    polygon.append(QPointF(x - labelWidth - 15, y - 4));
    polygon.append(QPointF(x - labelWidth - 15, y + 2));
    polygon.append(QPointF(x - labelWidth - 10, y - 1));
  }
  path.addPolygon(polygon);

  painter->fillPath(path, Qt::black);
}

bool MultiConnection::clickEvent(int x, int y) {
  QFontMetrics fontMetrics(QApplication::font());
  int labelWidth = fontMetrics.boundingRect(label).width();
  if (this->x - 20 - labelWidth <= x && this->y -10 <= y && y <= this->y + 10) {
    expanded = !expanded;
    parent->updateSize();
    parent->update();
    return true;
  }
  return false;
}

int MultiConnection::getHeight() const {
  // 16 is the connection distance
  if (expanded) {
    return connections.size() * parent->connectionDistance;
  } else {
    return parent->connectionDistance;
  }
}

ToolItem::ToolItem(const std::string& label, Workbench *bench)
 : QObject(), label(label), bench(bench), width(190), height(90), adjust(3 + 10), connectionDistance(16), inputsWidth(0),
   labelWidth(35), outputsWidth(0), labelFont(QApplication::font()), deletable(true),
   progress(Neutral)
{
  setFlag(ItemIsMovable);
  setFlag(ItemIsSelectable);
  setFlag(ItemIsFocusable);
#if (QT_VERSION >= 0x040700)
  setFlag(ItemSendsGeometryChanges);
#endif
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  labelFont.setBold(true);
  labelFont.setPointSize(10);

  updateSize();

//  QGraphicsDropShadowEffect *effect = new QGraphicsDropShadowEffect;
//  effect->setBlurRadius(8);
//  this->setGraphicsEffect(effect);
}

ToolItem::~ToolItem() {
  vector<ToolConnection*> tempIn(inputs);
  vector<MultiConnection*> tempOut(outputs);
  inputs.clear();
  outputs.clear();

  for (unsigned i = 0; i < tempIn.size(); ++i)
    delete tempIn[i];
  for (unsigned i = 0; i < tempOut.size(); ++i)
    delete tempOut[i];
}

void ToolItem::setWorkbench(Workbench* bench) {
  this->bench = bench;
}

bool ToolItem::isDeletable() const {
  return deletable;
}

void ToolItem::setDeletable(bool deletable) {
  this->deletable = deletable;
}

void ToolItem::setProgress(double progress) {
  this->progress = progress;
  update();
}

ToolConnection* ToolItem::hitConnection(int x, int y, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i]->hit(x, y))
        return inputs[i];
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i) {
      vector<ToolConnection*> connections;
      if (outputs[i]->hits(connections, x, y))
        return connections[0];
    }
  }
  return 0;
}

bool ToolItem::hitConnections(std::vector<ToolConnection*>& connections, int x, int y, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    ToolConnection* connection = hitConnection(x, y, direction);
    if (connection) {
      connections.push_back(connection);
      return true;
    }
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (outputs[i]->hits(connections, x, y))
        return true;
    }
  }
  return false;
}

ToolConnection* ToolItem::getConnection(int id, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i]->id == id)
        return inputs[i];
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i]->id == id)
        return outputs[i]->getLastConnection();
  }
  return 0;
}

QVariant ToolItem::itemChange(GraphicsItemChange change, const QVariant &value) {
  QPointF position = pos();
  switch (change) {
  case ItemPositionHasChanged:
    if (((int)position.x()) % 15 != 7 || ((int)position.y()) % 15 != 7) {
      position.setX((int)position.x() / 15 * 15 + 7);
      position.setY((int)position.y() / 15 * 15 + 7);
      setPos(position);
      return QGraphicsItem::itemChange(change, value);
    }
    updateCables();
    if (bench)
      bench->notifyItemChange(this);
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

void ToolItem::updateCables() {
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->cable)
      inputs[i]->cable->adjust();
  }
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputs[i]->adjust();
}

void ToolItem::updateSize() {
  QFontMetrics fontMetrics(QApplication::font());
  QFontMetrics labelFontMetrics(labelFont);
  inputsWidth = outputsWidth = 0;
  for (unsigned i = 0; i < inputs.size(); ++i)
    inputsWidth = max(inputsWidth, fontMetrics.boundingRect(inputs[i]->label).width() + 8);
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputsWidth = max(outputsWidth, fontMetrics.boundingRect(outputs[i]->getLabel()).width() + 14);

  width = (inputs.size() ? inputsWidth : 0) + labelWidth + (outputs.size() ? outputsWidth : 0);
  height = 30;
  height = max(height, labelFontMetrics.boundingRect(getLabel().c_str()).width() + 20);
  height = max(height, connectionDistance * (int)inputs.size() + 12);
  int outputsHeight = 12;
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputsHeight += outputs[i]->getHeight();
  height = max(height, outputsHeight);
  updateConnectionPositions();
}

void ToolItem::updateConnectionPositions() {
  for (int i = 0, pos = -((int)inputs.size()-1) * connectionDistance / 2 + height/2; i < (int)inputs.size(); ++i, pos += connectionDistance) {
    inputs[i]->setPos(0, pos);
  }

  int outputsHeight = 0;
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputsHeight += outputs[i]->getHeight();

  //for (int i = 0, pos = -((int)outputs.size()-1) * connectionDistance / 2 + height/2; i < (int)outputs.size(); ++i, pos += connectionDistance) {
  for (int i = 0, pos = -outputsHeight / 2 + connectionDistance / 2 + height/2; i < (int)outputs.size(); pos += outputs[i]->getHeight(), ++i) {
    outputs[i]->setPos(width, pos);
  }
  updateCables();
}

void ToolItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
  if (bench && bench->getCurrentItem() != this)
    bench->setCurrentItem(this);

  QPointF mousePos = mapToItem(this, event->pos());
  for (unsigned i = 0; i < outputs.size(); ++i) {
    if (outputs[i]->clickEvent(mousePos.x(), mousePos.y())) {
      return;
    }
  }
  QGraphicsItem::mousePressEvent(event);
}

void ToolItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
  QGraphicsItem::mouseReleaseEvent(event);
}

void ToolItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  Q_EMIT showDialogRequested(this);

  QGraphicsItem::mouseDoubleClickEvent(event);
}

bool ToolItem::isCurrentItem() const {
  return this == bench->getCurrentItem();
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
  QLinearGradient progressGradient(0, 0, 0, height);

  bool selected = bench->scene()->selectedItems().contains(this);

  qreal opacity = 1.0;

  if (bench && isCurrentItem()) {
    gradient.setColorAt(0, Qt::white);
    switch ((int)progress) {
    case -1:
      gradient.setColorAt(1, Qt::lightGray);
      break;
    case -2:
      gradient.setColorAt(1, Qt::yellow);
      break;
    default:
      gradient.setColorAt(1, Qt::red);
    }
    progressGradient.setColorAt(0, Qt::white);
    progressGradient.setColorAt(1, Qt::green);
    setZValue(7);
  } else {
    gradient.setColorAt(0, Qt::lightGray);
    switch ((int)progress) {
    case -1:
      gradient.setColorAt(1, Qt::gray);
      break;
    case -2:
      gradient.setColorAt(1, Qt::yellow);
      break;
    default:
      gradient.setColorAt(1, Qt::red);
    }
    progressGradient.setColorAt(0, Qt::lightGray);
    progressGradient.setColorAt(1, Qt::green);
    if (bench->isDependent(this)) {
      setZValue(5);
    } else {
      setZValue(2);
      painter->setOpacity(opacity = 0.5);
    }
  }

  painter->save();
  painter->setBrush(Qt::black);
 
  if (selected) {
    const int offset = 2;
    painter->setOpacity(0.1 * opacity);
    painter->setPen(QPen(Qt::black, 14));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.2 * opacity);
    painter->setPen(QPen(Qt::black, 9));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.25 * opacity);
    painter->setPen(QPen(Qt::black, 5.5));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.3 * opacity);
    painter->setPen(QPen(Qt::black, 2.5));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
  } else {
    /*const int offset = 1;
    painter->setOpacity(0.1);
    painter->setPen(QPen(Qt::black, 7));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.2);
    painter->setPen(QPen(Qt::black, 5));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.25);
    painter->setPen(QPen(Qt::black, 3));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);
    painter->setOpacity(0.3);
    painter->setPen(QPen(Qt::black, 1.5));
    painter->drawRoundedRect(0, offset, width, height, 4, 4);*/
  }
  painter->setOpacity(0.9 * opacity);
  painter->setBrush(gradient);
  painter->setPen(QPen(Qt::black, 0));
  painter->drawRoundedRect(0, 0, width, height, 4, 4);

  if (progress >=0) {
    painter->save();
    painter->setClipping(true);
    painter->setClipRect(QRectF(0, 0, width * min(100., progress) / 100., height));
    painter->setBrush(progressGradient);
    painter->drawRoundedRect(0, 0, width, height, 4, 4);
    painter->restore();
  }
  painter->restore();
}

std::string ToolItem::getLabel() const {
  return label;
}

void ToolItem::setLabel(const std::string& label) {
  this->label = label;
  updateSize();
  update();
}

void ToolItem::addConnection(const QString& label, int id, ToolConnection::Direction direction) {
  //ToolConnection* connection = new ToolConnection()
  if (direction == ToolConnection::Input) {
    inputs.push_back(new ToolConnection(label, direction, this, id));

  } else {
    outputs.push_back(new MultiConnection(label, direction, this, id));
  }
  updateConnectionPositions();
  updateSize();
  update();
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

std::vector<ToolConnection*>& ToolItem::getInputs() {
  return inputs;
}

void ToolItem::getOutputs(std::vector<ToolConnection*>& connections) {
  for (unsigned i = 0; i < outputs.size(); ++i) {
    MultiConnection* multi = outputs[i];
    for (unsigned j = 0; j < multi->connections.size(); ++j)
      connections.push_back(multi->connections[j]);
  }
}

}
