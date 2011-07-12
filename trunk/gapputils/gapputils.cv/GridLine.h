#pragma once
#ifndef GAPPUTILSCV_GRIDLINE_H_
#define GAPPUTILSCV_GRIDLINE_H_

#include <QGraphicsItem>
#include <QPointF>

namespace gapputils {

namespace cv {

class GridPointItem;
class GridWidget;

class GridLine : public QGraphicsItem
{
public:
  enum Orientation {Horizontal, Vertical};

private:
  GridPointItem *northWest, *southEast;
  QPointF sourcePoint, destPoint;
  Orientation orientation;
  GridWidget* parent;

public:
  GridLine(GridPointItem* northWest, GridPointItem* southEast, Orientation orientation, GridWidget* parent);
  virtual ~GridLine(void);

  void adjust();
  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

  GridPointItem* getNorthWest() const;
  GridPointItem* getSouthEast() const;
  Orientation getOrientation() const;
};

}

}

#endif /* GAPPUTILSCV_GRIDLINE_H_ */
