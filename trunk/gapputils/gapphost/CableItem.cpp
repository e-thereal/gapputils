#include "CableItem.h"

#include "ToolItem.h"
#include <qpainter.h>
#include "Workbench.h"
#include <iostream>

#include <qgraphicseffect.h>

using namespace capputils;
using namespace std;

namespace gapputils {

CableItem::CableItem(Workbench* bench,
    boost::shared_ptr<ToolConnection> input,
    boost::shared_ptr<ToolConnection> output) : bench(bench)
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

CableItem::~CableItem(void) { }

void CableItem::adjust() {
  QLineF line;
  boost::shared_ptr<ToolConnection> input = this->input.lock(), output = this->output.lock();
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
  dragPoint = boost::shared_ptr<QPointF>(new QPointF(point));
  adjust();
}

void CableItem::setInput(boost::shared_ptr<ToolConnection> input) {
  if (!this->input.expired())
    this->input.lock()->connect(0);
  this->input = input;
  if (input) {
    input->connect(this);
  }
  adjust();
}

void CableItem::setOutput(boost::shared_ptr<ToolConnection> output) {
  if (!this->output.expired())
    this->output.lock()->connect(0);
  this->output = output;
  if (output) {
    output->connect(this);
  }
  adjust();
}

boost::shared_ptr<ToolConnection> CableItem::getInput() const {
  return input.lock();
}

boost::shared_ptr<ToolConnection> CableItem::getOutput() const {
  return output.lock();
}

bool CableItem::needInput() const {
  return input.expired();
}

bool CableItem::needOutput() const {
  return output.expired();
}

void CableItem::endDrag() {
  dragPoint.reset();
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

  if (isCurrentCable || (bench->getCurrentCables().size() == 0  && !input.expired() && input.lock()->parent->isCurrentItem()) || (bench->getCurrentCables().size() == 0  && !output.expired() && output.lock()->parent->isCurrentItem())) {
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
