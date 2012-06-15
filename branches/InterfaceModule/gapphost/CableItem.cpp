#include "CableItem.h"

#include "ToolItem.h"
#include <qpainter.h>
#include "Workbench.h"
#include <iostream>

#include <qgraphicseffect.h>

using namespace capputils;
using namespace std;

namespace gapputils {

CableItem::CableItem(Workbench* bench, ToolConnection* input, ToolConnection* output) : input(0), output(0), dragPoint(0), bench(bench)
{
  setAcceptedMouseButtons(0);
  if (input)
    setInput(input);
  if (output)
    setOutput(output);
  adjust();
  setZValue(1);

//  QGraphicsDropShadowEffect *effect = new QGraphicsDropShadowEffect;
//  effect->setBlurRadius(8);
//  this->setGraphicsEffect(effect);
}

CableItem::~CableItem(void)
{
  if (dragPoint)
    delete dragPoint;
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
  if (this->input)
    this->input->connect(0);
  this->input = input;
  if (input) {
    input->connect(this);
  }
  adjust();
}

void CableItem::setOutput(ToolConnection* output) {
  if (this->output)
    this->output->connect(0);
  this->output = output;
  if (output) {
    output->connect(this);
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

  if (isCurrentCable || (bench->getCurrentCables().size() == 0  && input && input->parent->isCurrentItem()) || (bench->getCurrentCables().size() == 0  && output && output->parent->isCurrentItem())) {
    setZValue(6);
    painter->setPen(QPen(Qt::darkGray, 4.5));
    painter->drawPath(path);
    painter->setPen(QPen(Qt::white, 2.5));
    painter->drawPath(path);
  } else {
    if (bench->isDependent(this)) {
      setZValue(4);
      painter->setPen(QPen(Qt::black, 2.5));
      painter->drawPath(path);
    } else {
      setZValue(1);
      painter->setOpacity(0.25);
      painter->setPen(QPen(Qt::black, 2.5));
      painter->drawPath(path);
    }
  }
}

}
