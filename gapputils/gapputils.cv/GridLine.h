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
private:
  GridPointItem *fromItem, *toItem;
  QPointF sourcePoint, destPoint;
  GridWidget* parent;

public:
  GridLine(GridPointItem* fromItem, GridPointItem* toItem, GridWidget* parent);
  virtual ~GridLine(void);

  void adjust();
  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

}

}

#endif /* GAPPUTILSCV_GRIDLINE_H_ */
