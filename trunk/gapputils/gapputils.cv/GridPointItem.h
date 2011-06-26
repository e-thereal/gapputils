#pragma once
#ifndef GAPPUTILSCV_GRIDPOINTITEM_H_
#define GAPPUTILSCV_GRIDPOINTITEM_H_

#include <QGraphicsItem>

#include <vector>

#include "GridLine.h"
#include "GridPoint.h"

namespace gapputils {

namespace cv {

class GridPointItem : public QGraphicsItem
{
private:
  int radius, adjust;
  std::vector<GridLine*> lines;
  GridPoint* point;

public:
  GridPointItem(GridPoint* point);
  virtual ~GridPointItem(void);

  virtual QRectF boundingRect() const;
  virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  virtual QPainterPath shape() const;

  void addLine(GridLine* line);

private:
  void updateLines();
};

}

}

#endif /* GAPPUTILSCV_GRIDPOINTITEM_H_ */