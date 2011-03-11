#include "CableItem.h"

#include "ToolItem.h"
#include <qpainter.h>

namespace gapputils {

CableItem::CableItem(ToolConnection* input, ToolConnection* output) : input(input), output(output), dragPoint(0)
{
  setAcceptedMouseButtons(0);
  if (input)
    input->connect(this);
  if (output)
    output->connect(this);
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
  input->connect(this);
  adjust();
}

void CableItem::setOutput(ToolConnection* output) {
  this->output = output;
  output->connect(this);
  adjust();
}

bool CableItem::needInput() const {
  return !input;
}

bool CableItem::needOutput() const {
  return !output;
}

void CableItem::disconnectInput() {
  if (input) {
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
  
void CableItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  QPainterPath path;
  path.moveTo(sourcePoint);
  path.cubicTo(sourcePoint + QPointF(50, 0), destPoint + QPointF(-50, 0), destPoint);
  painter->setPen(QPen(Qt::black, 2.5));
  painter->drawPath(path);
}

}
