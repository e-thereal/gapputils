#include "GridPointItem.h"

#include <QPainter>

#include "GridModel.h"
#include "GridWidget.h"
#include <culib/math3d.h>

#include <iostream>

using namespace std;

namespace gapputils {

namespace cv {

GridPointItem::GridPointItem(GridPoint* point, GridModel* model, GridWidget* parent)
 : radius(4), adjust(2), smallRadius(3), northLine(0), southLine(0), westLine(0), eastLine(0),
   point(point), model(model), parent(parent)
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
    parent->startGridAdjustments();
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
  if (point->getFixed()) {
    path.addEllipse(-radius / scale, -radius / scale, 2 * radius / scale, 2 * radius / scale);
    painter->fillPath(path, Qt::white);
  } else {
    path.addEllipse(-smallRadius / scale, -smallRadius / scale, 2 * smallRadius / scale, 2 * smallRadius / scale);
    painter->fillPath(path, Qt::gray);
  }
  painter->drawPath(path);
}

void GridPointItem::addLine(GridLine* line) {
  if (line->getOrientation() == GridLine::Horizontal) {
    if (line->getNorthWest() == this)
      eastLine = line;
    else
      westLine = line;
  } else {
    if (line->getNorthWest() == this)
      southLine = line;
    else
      northLine = line;
  }
}

GridPointItem* GridPointItem::getNorth() const {
  if (northLine)
    return northLine->getNorthWest();
  return 0;
}

GridPointItem* GridPointItem::getSouth() const {
  if (southLine)
    return southLine->getSouthEast();
  return 0;
}

GridPointItem* GridPointItem::getWest() const {
  if (westLine)
    return westLine->getNorthWest();
  return 0;
}

GridPointItem* GridPointItem::getEast() const {
  if (eastLine)
    return eastLine->getSouthEast();
  return 0;
}

void GridPointItem::calculateForces() {
  if (!scene() || point->getFixed()) {
    newPos = pos();
    return;
  }

  double xvel = 0.0, yvel = 0.0;

  double weight = 3;

  GridPointItem *n = getNorth(), *s = getSouth(), *e = getEast(), *w = getWest();
  float2 np = (n ? make_float2(mapToItem(n, 0, 0).x(), mapToItem(n, 0, 0).y()) : make_float2(0, 0));
  float2 sp = (s ? make_float2(mapToItem(s, 0, 0).x(), mapToItem(s, 0, 0).y()) : make_float2(0, 0));
  float2 ep = (e ? make_float2(mapToItem(e, 0, 0).x(), mapToItem(e, 0, 0).y()) : make_float2(0, 0));
  float2 wp = (w ? make_float2(mapToItem(w, 0, 0).x(), mapToItem(w, 0, 0).y()) : make_float2(0, 0));

  float2 target = make_float2(0, 0);

  if (n && s && e && w) {

    float2 nwp = make_float2(mapToItem(n->getWest(), 0, 0).x(), mapToItem(n->getWest(), 0, 0).y());
    float2 nep = make_float2(mapToItem(n->getEast(), 0, 0).x(), mapToItem(n->getEast(), 0, 0).y());
    float2 swp = make_float2(mapToItem(s->getWest(), 0, 0).x(), mapToItem(s->getWest(), 0, 0).y());
    float2 sep = make_float2(mapToItem(s->getEast(), 0, 0).x(), mapToItem(s->getEast(), 0, 0).y());

    float2 nvh = -nwp + nep;
    float2 svh = -swp + sep;
    float nhratio = dot(nvh, np - nwp) / dot(nvh, nvh);
    float shratio = dot(svh, sp - swp) / dot(svh, svh);

    float2 vh = -wp + ep;
    vh = (dot(vh, wp) / dot(vh, vh) + 0.5 * (nhratio + shratio)) * vh;

    float2 wvv = -nwp + swp;
    float2 evv = -nep + sep;
    float wvratio = dot(wvv, wp - nwp) / dot(wvv, wvv);
    float evratio = dot(evv, ep - nep) / dot(evv, evv);

    float2 vv = -np + sp;
    vv = (dot(vv, np) / dot(vv, vv) + 0.5 * (wvratio + evratio)) * vv;

    target = vv + vh;
  } else if (n && s) {
    target = 0.5 * np + 0.5 * sp;
  } else if (w && e) {
    target = 0.5 * wp + 0.5 * ep;
  }
  xvel = -target.x / weight;
  yvel = -target.y / weight;

  if (fabs(xvel) < 0.01 && fabs(yvel) < 0.01) {
    xvel = 0;
    yvel = 0;
  }
  newPos = pos() + QPointF(xvel, yvel);
}

bool GridPointItem::advance() {
  if (newPos == pos())
    return false;

  setPos(newPos);
  return true;
}

void GridPointItem::updateLines() {
  if (northLine)
    northLine->adjust();
  if (southLine)
    southLine->adjust();
  if (westLine)
    westLine->adjust();
  if (eastLine)
    eastLine->adjust();
}

void GridPointItem::mousePressEvent(QGraphicsSceneMouseEvent* e) {
  point->setFixed(true);
  update();

  QGraphicsItem::mousePressEvent(e);
}

void GridPointItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  point->setFixed(false);
  update();
  parent->startGridAdjustments();

  // For some reason, calling mouseDoubleClick will set the point to fixed again.
  //QGraphicsItem::mouseDoubleClickEvent(event);
}

}

}
