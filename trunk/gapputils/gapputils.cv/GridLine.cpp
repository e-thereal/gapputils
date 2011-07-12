#include "GridLine.h"

#include <QPainter>

#include "GridPointItem.h"
#include "GridWidget.h"

namespace gapputils {

namespace cv {

GridLine::GridLine(GridPointItem* northWest, GridPointItem* southEast, Orientation orientation, GridWidget* parent)
 : northWest(northWest), southEast(southEast), orientation(orientation), parent(parent)
{
  northWest->addLine(this);
  southEast->addLine(this);

  setAcceptedMouseButtons(0);
  adjust();
}

GridLine::~GridLine(void)
{
}

void GridLine::adjust() {
  sourcePoint = mapFromScene(QPointF(northWest->x(), northWest->y()));
  destPoint = mapFromScene(QPointF(southEast->x(), southEast->y()));

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

GridPointItem* GridLine::getNorthWest() const {
  return northWest;
}

GridPointItem* GridLine::getSouthEast() const {
  return southEast;
}

GridLine::Orientation GridLine::getOrientation() const {
  return orientation;
}

}

}
