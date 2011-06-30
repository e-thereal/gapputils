#include "GridLine.h"

#include <QPainter>

#include "GridPointItem.h"
#include "GridWidget.h"

namespace gapputils {

namespace cv {

GridLine::GridLine(GridPointItem* fromItem, GridPointItem* toItem, GridWidget* parent)
 : fromItem(fromItem), toItem(toItem), parent(parent)
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
  qreal scale = parent->getViewScale();
  painter->setPen(QPen(Qt::black, 2. / scale));
  painter->drawLine(sourcePoint, destPoint);
  painter->setPen(QPen(Qt::white, 1. / scale));
  painter->drawLine(sourcePoint, destPoint);
}

}

}
