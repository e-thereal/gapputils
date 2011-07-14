/*
 * RectangleItem.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "RectangleItem.h"

#include <QPainter>

#include "RectangleModel.h"
#include "RectangleWidget.h"

namespace gapputils {

namespace cv {

RectangleItem::RectangleItem(boost::shared_ptr<RectangleModel> model, RectangleWidget* parent)
 : model(model), parent(parent), handler(this, &RectangleItem::changedHandler)
{
  setFlag(ItemIsMovable);
  setFlag(ItemIsSelectable);
  setFlag(ItemIsFocusable);
  setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(3);

  setPos(QPointF(model->getLeft(), model->getTop()));

  model->Changed.connect(handler);
}

RectangleItem::~RectangleItem() {
  model->Changed.disconnect(handler);
}

void RectangleItem::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == RectangleModel::leftId)
    setX(model->getLeft());
  else if (eventId == RectangleModel::topId)
    setY(model->getTop());
  prepareGeometryChange();
}

QVariant RectangleItem::itemChange(GraphicsItemChange change, const QVariant &value) {
  switch (change) {
  case ItemPositionHasChanged:
    model->setLeft(x());
    model->setTop(y());
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

QRectF RectangleItem::boundingRect() const
{
  const float adjust = 3;
  return QRectF(-adjust, -adjust, model->getWidth() + 2 * adjust, model->getHeight() + 2 * adjust);
}

QPainterPath RectangleItem::shape() const {
  QPainterPath path;
  path.addRect(0, 0, model->getWidth(), model->getHeight());
  return path;
}

void RectangleItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
  Q_UNUSED(option)

  //painter->drawEllipse(0, 0, radius/2, radius/2);
  painter->setPen(QPen(Qt::black, 2. / parent->getViewScale()));
  painter->drawRect(0, 0, model->getWidth(), model->getHeight());
  painter->setPen(QPen(Qt::white, 1. / parent->getViewScale()));
  painter->drawRect(0, 0, model->getWidth(), model->getHeight());
}

}

}
