#pragma once
#ifndef GAPPUTILSCV_GRIDPOINTITEM_H_
#define GAPPUTILSCV_GRIDPOINTITEM_H_

#include <QGraphicsItem>

#include <vector>

#include "GridLine.h"
#include "GridPoint.h"

namespace gapputils {

namespace cv {

class GridModel;

class GridPointItem : public QGraphicsItem
{
private:
  int radius, adjust;
  std::vector<GridLine*> lines;
  GridPoint* point;
  GridModel* model;

public:
  GridPointItem(GridPoint* point, GridModel* model);
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