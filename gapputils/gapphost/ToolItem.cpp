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
#include <qapplication.h>
#include <QFontMetrics>
#include <qstylepainter.h>
#include "Workbench.h"
#include <qtimer.h>

#include <qpicture.h>
#include "CableItem.h"
#include <qgraphicseffect.h>

using namespace std;

namespace gapputils {

using namespace workflow;

ToolConnection::ToolConnection(const QString& label, Direction direction,
    ToolItem* parent, const std::string& id, MultiConnection* multi)
  : x(0), y(0), width(6), height(7), label(label), direction(direction),
    parent(parent), id(id), cable(0), multi(multi)
{
}

ToolConnection::~ToolConnection() { }

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
        parent->bench->areCompatible(currentCable->getInput().get(), this))
    {
      painter->setBrush(Qt::white);
      painter->setOpacity(1.0);
//      painter->setFont(boldFont);
    } else if(parent->isCurrentItem() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getOutput().get() == this) {
      painter->setBrush(Qt::white);
    } else if (!currentCable && cable && cable->getInput() && cable->getInput()->parent && cable->getInput()->parent->isCurrentItem()) {
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
        parent->bench->areCompatible(this, currentCable->getOutput().get()))
    {
      painter->setBrush(Qt::white);
      painter->setOpacity(1.0);
//      painter->setFont(boldFont);
    } else if(parent->isCurrentItem() && !currentCable) {
      painter->setBrush(Qt::white);
    } else if (currentCable && currentCable->getInput().get() == this) {
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

void ToolConnection::setLabel(const QString& label) {
  if (multi) {
    multi->label = label;
    for (unsigned i = 0; i < multi->connections.size(); ++i)
      multi->connections[i]->label = label;
  } else {
    this->label = label;
  }
  parent->updateSize();
  parent->update();
}

int ToolConnection::getIndex() const {
  if (!multi)
    return 0;

  for (size_t i = 0; i < multi->connections.size(); ++i) {
    if (multi->connections[i].get() == this) {
      return i;
    }
  }

  assert(0);
}

MultiConnection::MultiConnection(const QString& label, ToolConnection::Direction direction,
    ToolItem* parent, const std::string& id, bool staticConnection)
 : label(label), direction(direction), parent(parent), id(id), expanded(false), staticConnection(staticConnection)
{
  connections.push_back(boost::shared_ptr<ToolConnection>(new ToolConnection(label, direction, parent, id, this)));
}

MultiConnection::~MultiConnection() { }

bool MultiConnection::hits(std::vector<boost::shared_ptr<ToolConnection> >& connections,
    int x, int y) const
{
  for (unsigned i = 0; i < this->connections.size(); ++i)
    if (this->connections[i]->hit(x, y))
      connections.push_back(this->connections[i]);
  return connections.size();
}

// TODO: depreciated. will be replaced by getConnections() or new connection
boost::shared_ptr<ToolConnection> MultiConnection::getLastConnection() {
  if (connections.size())
    return connections[connections.size() - 1];
  return boost::shared_ptr<ToolConnection>();
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
  if (staticConnection)
    return;

  for (size_t i = 0; i < connections.size() - 1; ++i) {
    if (connections[i]->cable == 0) {
      connections.erase(connections.begin() + i);
      --i;
    }
  }
  if (connections.size() && connections[connections.size() - 1]->cable)
    connections.push_back(boost::shared_ptr<ToolConnection>(new ToolConnection(label, direction, parent, id, this)));
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
  for (unsigned i = 0; i < connections.size() && (expanded ? true : i < 1); ++i)
    connections[i]->draw(painter, showLabel);

  if (!staticConnection) {
    QPainterPath path;
    QPolygonF polygon;
    if (direction == ToolConnection::Input) {
      if (expanded) {
        polygon.append(QPointF(x + labelWidth + 16, y - 2));
        polygon.append(QPointF(x + labelWidth + 10, y - 2));
        polygon.append(QPointF(x + labelWidth + 13, y + 3));
      } else {
        polygon.append(QPointF(x + labelWidth + 15, y - 3));
        polygon.append(QPointF(x + labelWidth + 15, y + 3));
        polygon.append(QPointF(x + labelWidth + 10, y - 0));
      }
    } else {
      if (expanded) {
        polygon.append(QPointF(x - labelWidth - 16, y - 2));
        polygon.append(QPointF(x - labelWidth - 10, y - 2));
        polygon.append(QPointF(x - labelWidth - 13, y + 3));
      } else {
        polygon.append(QPointF(x - labelWidth - 15, y - 3));
        polygon.append(QPointF(x - labelWidth - 15, y + 3));
        polygon.append(QPointF(x - labelWidth - 10, y - 0));
      }
    }
    path.addPolygon(polygon);
    painter->fillPath(path, Qt::black);
  }
}

bool MultiConnection::clickEvent(int x, int y) {
  QFontMetrics fontMetrics(QApplication::font());
  int labelWidth = fontMetrics.boundingRect(label).width();
  if (direction == ToolConnection::Input) {
    if (x <= this->x + 20 + labelWidth && this->y -10 <= y && y <= this->y + 10) {
      expanded = !expanded;
      parent->updateSize();
      parent->update();
      return true;
    }
  } else {
    if (this->x - 20 - labelWidth <= x && this->y -10 <= y && y <= this->y + 10) {
      expanded = !expanded;
      parent->updateSize();
      parent->update();
      return true;
    }
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

ToolItem::ToolItem(const std::string& label, Workbench* bench)
 : QObject(), label(label), bench(bench), width(190), height(90), adjust(3 + 10), connectionDistance(16), inputsWidth(0),
   labelWidth(35), outputsWidth(0), labelFont(QApplication::font()), progress(Neutral), itemStyle(Normal),
   doubleClicked(false)
{
  setFlag(ItemIsMovable);
  setFlag(ItemIsSelectable);
  setFlag(ItemIsFocusable);
#if (QT_VERSION >= 0x040700)
  setFlag(ItemSendsGeometryChanges);
#endif
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  effect = new QGraphicsDropShadowEffect;
  setGraphicsEffect(effect);

  if (itemStyle == Normal) {
    labelFont.setBold(true);
    labelFont.setPointSize(10);
    effect->setColor(QColor(0, 0, 0, 160));
    effect->setEnabled(false);
  } else if (itemStyle == MessageBox ) {
    labelFont.setBold(false);
    labelFont.setPointSize(12);
    labelWidth = 36;
    effect->setColor(QColor(0, 0, 0, 72));
    effect->setEnabled(true);
  } else {
    labelFont.setBold(false);
    labelFont.setPointSize(14);
    labelWidth = 36;
    effect->setColor(QColor(0, 0, 0, 72));
    effect->setEnabled(true);
  }

  updateSize();
}

ToolItem::~ToolItem() { }

void ToolItem::setItemStyle(ItemStyle style) {
  itemStyle = style;
  if (itemStyle == Normal) {
    labelFont.setBold(true);
    labelFont.setPointSize(10);
    this->setGraphicsEffect(0);
    effect->setColor(QColor(0, 0, 0, 160));
    effect->setEnabled(false);
  } else if (itemStyle == MessageBox ) {
    labelFont.setBold(false);
    labelFont.setPointSize(12);
    labelWidth = 36;
    effect->setColor(QColor(0, 0, 0, 72));
    effect->setEnabled(true);
  } else {
    labelFont.setBold(false);
    labelFont.setPointSize(14);
    labelWidth = 36;
    effect->setColor(QColor(0, 0, 0, 72));
    effect->setEnabled(true);
  }
  updateSize();
  update();
}

void ToolItem::setWorkbench(Workbench* bench) {
  this->bench = bench;
}

void ToolItem::setProgress(double progress) {
  this->progress = progress;
  update();
}

boost::shared_ptr<ToolConnection> ToolItem::hitConnection(int x, int y, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i) {
      vector<boost::shared_ptr<ToolConnection> > connections;
      if (inputs[i]->hits(connections, x, y))
        return connections[0];
    }
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i) {
      vector<boost::shared_ptr<ToolConnection> > connections;
      if (outputs[i]->hits(connections, x, y))
        return connections[0];
    }
  }
  return boost::shared_ptr<ToolConnection>();
}

bool ToolItem::hitConnections(std::vector<boost::shared_ptr<ToolConnection> >& connections, int x, int y, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->hits(connections, x, y))
        return true;
    }

//    boost::shared_ptr<ToolConnection> connection = hitConnection(x, y, direction);
//    if (connection) {
//      connections.push_back(connection);
//      return true;
//    }
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (outputs[i]->hits(connections, x, y))
        return true;
    }
  }
  return false;
}

boost::shared_ptr<ToolConnection> ToolItem::getConnection(const std::string& id, ToolConnection::Direction direction) const {
  if (direction == ToolConnection::Input) {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i]->id == id)
        return inputs[i]->getLastConnection();
  } else {
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i]->id == id)
        return outputs[i]->getLastConnection();
  }
  return boost::shared_ptr<ToolConnection>();
}

QVariant ToolItem::itemChange(GraphicsItemChange change, const QVariant &value) {
  QPointF position = pos();
  int snapDistance = 15;

  switch (change) {
  case ItemPositionHasChanged:

    if (((int)position.x()) % snapDistance || ((int)position.y()) % snapDistance) {
      position.setX(((int)position.x() + snapDistance / 2) / snapDistance * snapDistance);
      position.setY(((int)position.y() + snapDistance / 2) / snapDistance * snapDistance);
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
  for (unsigned i = 0; i < inputs.size(); ++i)
    inputs[i]->adjust();
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputs[i]->adjust();
}

void ToolItem::updateSize() {
  QFontMetrics fontMetrics(QApplication::font());
  QFontMetrics labelFontMetrics(labelFont);

  if (itemStyle == Normal) {
    inputsWidth = outputsWidth = 0;
    for (unsigned i = 0; i < inputs.size(); ++i)
      inputsWidth = max(inputsWidth, fontMetrics.boundingRect(inputs[i]->getLabel()).width() + (inputs[i]->staticConnection ? 8 : 14));
    for (unsigned i = 0; i < outputs.size(); ++i)
      outputsWidth = max(outputsWidth, fontMetrics.boundingRect(outputs[i]->getLabel()).width() + 14);

    width = (inputs.size() ? inputsWidth : 0) + labelWidth + (outputs.size() ? outputsWidth : 0);

    // minimum height
    height = 30;

    // at least as high as the label
    height = max(height, labelFontMetrics.boundingRect(getLabel().c_str()).width() + 20);

    // at least as high as the inputs
    int inputsHeight = 12;
    for (size_t i = 0; i < inputs.size(); ++i)
      inputsHeight += inputs[i]->getHeight();
    height = max(height, inputsHeight);

    // at least as high as the outputs
    int outputsHeight = 12;
    for (unsigned i = 0; i < outputs.size(); ++i)
      outputsHeight += outputs[i]->getHeight();
    height = max(height, outputsHeight);

    updateConnectionPositions();
  } else if (itemStyle == HorizontalAnnotation ||itemStyle == MessageBox) {
    width = labelFontMetrics.boundingRect(getLabel().c_str()).width() + 32;
    height = labelWidth;
  } else if (itemStyle == VerticalAnnotation) {
    width = labelWidth;
    height = labelFontMetrics.boundingRect(getLabel().c_str()).width() + 32;
  }
}

void ToolItem::updateConnectionPositions() {
//  for (int i = 0, pos = -((int)inputs.size()-1) * connectionDistance / 2 + height/2; i < (int)inputs.size(); ++i, pos += connectionDistance) {
//    inputs[i]->setPos(0, pos);
//  }

  int left = -width / 2.0, top = -height / 2.0;

  int inputsHeight = 0;
  for (size_t i = 0; i < inputs.size(); ++i)
    inputsHeight += inputs[i]->getHeight();

  for (int i = 0, pos = -inputsHeight / 2 + connectionDistance / 2 + height / 2 + top; i < (int)inputs.size(); pos += inputs[i]->getHeight(), ++i) {
    inputs[i]->setPos(left, pos);
  }

  int outputsHeight = 0;
  for (size_t i = 0; i < outputs.size(); ++i)
    outputsHeight += outputs[i]->getHeight();

  for (int i = 0, pos = -outputsHeight / 2 + connectionDistance / 2 + height / 2 + top; i < (int)outputs.size(); pos += outputs[i]->getHeight(), ++i) {
    outputs[i]->setPos(width + left, pos);
  }
  updateCables();
}

void ToolItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
  if (bench && bench->getCurrentItem() != this)
    bench->setCurrentItem(this);

  QPointF mousePos = mapToItem(this, event->pos());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->clickEvent(mousePos.x(), mousePos.y())) {
      return;
    }
  }
  for (unsigned i = 0; i < outputs.size(); ++i) {
    if (outputs[i]->clickEvent(mousePos.x(), mousePos.y())) {
      return;
    }
  }
  QGraphicsItem::mousePressEvent(event);
}

void ToolItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
  if (doubleClicked)
    Q_EMIT showDialogRequested(this);
  doubleClicked = false;
  QGraphicsItem::mouseReleaseEvent(event);
}

void ToolItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  doubleClicked = true;
  QGraphicsItem::mouseDoubleClickEvent(event);
}

bool ToolItem::isCurrentItem() const {
  return this == bench->getCurrentItem();
}

QRectF ToolItem::boundingRect() const
{
  int left = -width / 2.0, top = -height / 2.0;
  return QRectF(left - adjust, top - adjust, width+2*adjust, height+2*adjust);
}

QPainterPath ToolItem::shape() const {
  int left = -width / 2.0, top = -height / 2.0;

  QPainterPath path;
  path.addRect(left, top, width, height);
  return path;
}

void ToolItem::drawConnections(QPainter* painter, bool showLabel) {
  for (unsigned i = 0; i < inputs.size(); ++i)
    inputs[i]->draw(painter, showLabel);
  for (unsigned i = 0; i < outputs.size(); ++i)
    outputs[i]->draw(painter, showLabel);
}

void ToolItem::drawBox(QPainter* painter) {

  painter->save();

  int left = -width / 2.0, top = -height / 2.0;

  if (itemStyle == Normal) {

    QLinearGradient gradient(0, top, 0, height + top);
    QLinearGradient progressGradient(0, top, 0, height + top);

    bool selected = bench->scene()->selectedItems().contains(this);

    qreal opacity = 1.0;

    if (bench && isCurrentItem()) {
      gradient.setColorAt(0, Qt::white);
      switch ((int)progress) {
      case Neutral:
        gradient.setColorAt(1, Qt::lightGray);
        break;
      case InProgress:
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
      case Neutral:
        gradient.setColorAt(1, Qt::gray);
        break;
      case InProgress:
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

    painter->setBrush(Qt::black);

    effect->setEnabled(selected);

    painter->setOpacity(0.9 * opacity);
    painter->setBrush(gradient);
    painter->setPen(QPen(Qt::black, 0));
    painter->drawRoundedRect(left, top, width, height, 4, 4);

    if (progress >=0) {
      painter->save();
      painter->setClipping(true);
      painter->setClipRect(QRectF(left, top, width * min(100., progress) / 100., height));
      painter->setBrush(progressGradient);
      painter->drawRoundedRect(left, top, width, height, 4, 4);
      painter->restore();
    }
  } else if (itemStyle == MessageBox) {
    painter->setBrush(QColor(224, 224, 255));
    painter->setPen(QColor(96, 96, 64));
    painter->drawRect(left, top, width, height);
  } else {
    painter->setBrush(QColor(255, 255, 192));
    painter->setPen(QColor(96, 96, 64));
    painter->drawRect(left, top, width, height);
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

void ToolItem::addConnection(const QString& label, const std::string& id,
    ToolConnection::Direction direction, bool staticConnection)
{
  if (direction == ToolConnection::Input) {
    inputs.push_back(boost::shared_ptr<MultiConnection>(new MultiConnection(label, direction, this, id, staticConnection)));
  } else {
    outputs.push_back(boost::shared_ptr<MultiConnection>(new MultiConnection(label, direction, this, id, staticConnection)));
  }
  updateConnectionPositions();
  updateSize();
  update();
}

void ToolItem::deleteConnection(const std::string& id, ToolConnection::Direction direction) {
  if (direction == ToolConnection::Input) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->id == id) {
        std::vector<boost::shared_ptr<ToolConnection> >& connections = inputs[i]->connections;
        for (size_t j = 0; j < connections.size(); ++j) {
          if (connections[j]->cable)
            bench->removeCableItem(connections[j]->cable);
        }
        inputs.erase(inputs.begin() + i);
        break;
      }
    }
  } else {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i]->id == id) {
        std::vector<boost::shared_ptr<ToolConnection> >& connections = outputs[i]->connections;
        for (size_t j = 0; j < connections.size(); ++j) {
          if (connections[j]->cable)
            bench->removeCableItem(connections[j]->cable);
        }
        outputs.erase(outputs.begin() + i);
        break;
      }
    }
  }
  updateConnectionPositions();
  updateSize();
  update();
}

void ToolItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
  Q_UNUSED(option);

  int left = -width / 2.0, top = -height / 2.0;

  if (bench && effect) {
    effect->setBlurRadius(20 * bench->getViewScale());
    effect->setOffset(5 * bench->getViewScale());
  }

  QString label = getLabel().c_str();

  drawBox(painter);
  drawConnections(painter);
 
  painter->save();
  painter->setFont(labelFont);
  if (itemStyle == Normal) {
    painter->translate((inputs.size() ? inputsWidth : 0) + labelWidth/2 + left, height/2 + top);
    painter->rotate(270);
    painter->translate(-(inputs.size() ? inputsWidth : 0) -labelWidth/2, -height/2);
    painter->drawText((inputs.size() ? inputsWidth : 0) - height, 0, labelWidth + 2 * height, height, Qt::AlignCenter, label);
  } else if (itemStyle == HorizontalAnnotation || itemStyle == MessageBox) {
    painter->drawText(left, top, width, height, Qt::AlignCenter, label);
  } else if (itemStyle == VerticalAnnotation) {
    painter->translate(width/2 + left, height/2 + top);
    painter->rotate(270);
    painter->translate(-width/2, -height/2);
    painter->drawText(- height, 0, width + 2 * height, height, Qt::AlignCenter, label);
  }
  painter->restore();
}

void ToolItem::getInputs(std::vector<boost::shared_ptr<ToolConnection> >& connections) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    boost::shared_ptr<MultiConnection> multi = inputs[i];
    for (size_t j = 0; j < multi->connections.size(); ++j)
      connections.push_back(multi->connections[j]);
  }
}

void ToolItem::getOutputs(std::vector<boost::shared_ptr<ToolConnection> >& connections) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    boost::shared_ptr<MultiConnection> multi = outputs[i];
    for (size_t j = 0; j < multi->connections.size(); ++j)
      connections.push_back(multi->connections[j]);
  }
}

}
