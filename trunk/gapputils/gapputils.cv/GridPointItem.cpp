#include "GridPointItem.h"

#include <QPainter>

#include "GridModel.h"
#include "GridWidget.h"

namespace gapputils {

namespace cv {

GridPointItem::GridPointItem(GridPoint* point, GridModel* model, GridWidget* parent)
 : radius(4), adjust(2), point(point), model(model), parent(parent)
{
  setFlag(ItemIsMovable);
  setFlag(ItemIsSelectable);
  setFlag(ItemIsFocusable);
  setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  setPos(QPointF(point->getX(), point->getY()));
}

GridPointItem::~GridPointItem(void)
{
}

QVariant GridPointItem::itemChange(GraphicsItemChange change, const QVariant &value) {
  switch (change) {
  case ItemPositionHasChanged:
    updateLines();
    point->setX(x());
    point->setY(y());
    model->fireChangeEvent(GridModel::pointsId);
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

QRectF GridPointItem::boundingRect() const
{
  qreal scale = parent->getViewScale();
  return QRectF(-adjust - radius / scale, -adjust - radius / scale, 2 * radius / scale + 2 * adjust, 2 * radius / scale + 2 * adjust);
}

QPainterPath GridPointItem::shape() const {
  QPainterPath path;
  qreal scale = parent->getViewScale();
  path.addEllipse(-radius / scale, -radius / scale, 2 * radius / scale, 2 * radius / scale);
  return path;
}

void GridPointItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
  qreal scale = parent->getViewScale();
  Q_UNUSED(option)

  //painter->drawEllipse(0, 0, radius/2, radius/2);
  QPainterPath path;
  path.addEllipse(-radius / scale, -radius / scale, 2 * radius / scale, 2 * radius / scale);
  painter->fillPath(path, Qt::white);
  painter->drawPath(path);
}

void GridPointItem::addLine(GridLine* line) {
  lines.push_back(line);
}

void GridPointItem::updateLines() {
  for (unsigned i = 0; i < lines.size(); ++i)
    lines[i]->adjust();
}

}

}
