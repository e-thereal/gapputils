#include "CableItem.h"

#include "ToolItem.h"
#include <qpainter.h>
#include "Workbench.h"
#include <capputils/ObservableClass.h>
#include <iostream>

using namespace capputils;
using namespace std;

namespace gapputils {

void CableItem::ChangeEventHandler::operator()(ObservableClass*, int eventId) {
  if (!cableItem->input || !cableItem->output)
    return;

  if (eventId == cableItem->input->propertyId)
    cableItem->output->property->setValue(*cableItem->output->parent->getNode()->getModule(), *cableItem->input->parent->getNode()->getModule(), cableItem->input->property);
}

bool CableItem::ChangeEventHandler::operator==(const ChangeEventHandler& handler) const {
  return handler.cableItem == cableItem;
}

CableItem::CableItem(Workbench* bench, ToolConnection* input, ToolConnection* output) : changeEventHandler(this), input(0), output(0), dragPoint(0), bench(bench)
{
  setAcceptedMouseButtons(0);
  if (input)
    setInput(input);
  if (output)
    setOutput(output);
  adjust();
  setZValue(1);
}

CableItem::~CableItem(void)
{
  if (dragPoint)
    delete dragPoint;
  disconnectInput();
  disconnectOutput();
}

void CableItem::adjust() {
  QLineF line;
  if (input && output)
    line = QLineF(mapFromItem(input->parent, input->attachmentPos()), mapFromItem(output->parent, output->attachmentPos()));
  else if (input && dragPoint)
    line = QLineF(mapFromItem(input->parent, input->attachmentPos()), mapFromScene(*dragPoint));
  else if (dragPoint && output)
    line = QLineF(mapFromScene(*dragPoint), mapFromItem(output->parent, output->attachmentPos()));
  prepareGeometryChange();
  sourcePoint = line.p1();
  destPoint = line.p2();
}

void CableItem::setDragPoint(QPointF point) {
  if (dragPoint)
    delete dragPoint;
  dragPoint = new QPointF(point);
  adjust();
}

void CableItem::setInput(ToolConnection* input) {
  this->input = input;
  if (input) {
    input->connect(this);
    ObservableClass* observable = dynamic_cast<ObservableClass*>(input->parent->getNode()->getModule());
    if (observable) {
      observable->Changed.connect(changeEventHandler);
      //observable->fireChangeEvent(input->propertyId);
    }
    if (output) {
      output->property->setValue(*output->parent->getNode()->getModule(), *input->parent->getNode()->getModule(), input->property);
    }
  }
  adjust();
}

void CableItem::setOutput(ToolConnection* output) {
  this->output = output;
  if (output) {
    output->connect(this);
    if (input) {
      output->property->setValue(*output->parent->getNode()->getModule(), *input->parent->getNode()->getModule(), input->property);
    }
  }
  adjust();
}

ToolConnection* CableItem::getInput() const {
  return input;
}

ToolConnection* CableItem::getOutput() const {
  return output;
}

bool CableItem::needInput() const {
  return !input;
}

bool CableItem::needOutput() const {
  return !output;
}

void CableItem::disconnectInput() {
  if (input) {
    if (input->parent->getNode()) {
      ObservableClass* observable = dynamic_cast<ObservableClass*>(input->parent->getNode()->getModule());
      if (observable)
        observable->Changed.disconnect(changeEventHandler);
    }
    ToolConnection* temp = input;
    input = 0;
    temp->disconnect();
  }
}
 
void CableItem::disconnectOutput() {
  if (output) {
    ToolConnection* temp = output;
    output = 0;
    temp->disconnect();
  }
}

void CableItem::endDrag() {
  if (dragPoint)
    delete dragPoint;
  dragPoint = 0;
}

QRectF CableItem::boundingRect() const {
  return QRectF(sourcePoint, QSizeF(destPoint.x() - sourcePoint.x(),
    destPoint.y() - sourcePoint.y())).normalized().adjusted(-50, -50, 50, 50);
}
  
void CableItem::paint(QPainter *painter, const QStyleOptionGraphicsItem* /*option*/, QWidget* /*widget*/) {
  QPainterPath path;
  path.moveTo(sourcePoint);
  path.cubicTo(sourcePoint + QPointF(50, 0), destPoint + QPointF(-50, 0), destPoint);

  bool isCurrentCable = false;
  vector<CableItem*>& cables = bench->getCurrentCables();
  for (unsigned i = 0; i < cables.size(); ++i) {
    if (cables[i] == this) {
      isCurrentCable = true;
      break;
    }
  }

  if (isCurrentCable || (bench->getCurrentCables().size() == 0  && input && input->parent->isSelected()) || (bench->getCurrentCables().size() == 0  && output && output->parent->isSelected())) {
    setZValue(3);
    painter->setPen(QPen(Qt::darkGray, 4.5));
    painter->drawPath(path);
    painter->setPen(QPen(Qt::white, 2.5));
    painter->drawPath(path);
  } else {
    setZValue(1);
    painter->setPen(QPen(Qt::black, 2.5));
    painter->drawPath(path);
  }
}

}
