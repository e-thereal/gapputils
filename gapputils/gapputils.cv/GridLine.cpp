#include "GridLine.h"

#include <QPainter>

#include "GridPointItem.h"

namespace gapputils {

namespace cv {

GridLine::GridLine(GridPointItem* fromItem, GridPointItem* toItem) : fromItem(fromItem), toItem(toItem)
{
  fromItem->addLine(this);
  toItem->addLine(this);

  setAcceptedMouseButtons(0);
  adjust();
}

GridLine::~GridLine(void)
{
}

void GridLine::adjust() {
  sourcePoint = mapFromScene(QPointF(fromItem->x(), fromItem->y()));
  destPoint = mapFromScene(QPointF(toItem->x(), toItem->y()));

  prepareGeometryChange();
}

QRectF GridLine::boundingRect() const {
  return QRectF(sourcePoint, QSizeF(destPoint.x() - sourcePoint.x(),
    destPoint.y() - sourcePoint.y())).normalized().adjusted(-5, -5, 5, 5);
}

void GridLine::paint(QPainter *painter, const QStyleOptionGraphicsItem* /*option*/, QWidget* /*widget*/) {
  painter->setPen(QPen(Qt::black, 2));
  painter->drawLine(sourcePoint, destPoint);
  painter->setPen(QPen(Qt::white, 1));
  painter->drawLine(sourcePoint, destPoint);
}

}

}
