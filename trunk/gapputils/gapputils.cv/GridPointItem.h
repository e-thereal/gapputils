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
class GridWidget;

class GridPointItem : public QGraphicsItem
{
private:
  int radius, adjust, smallRadius;
  GridLine *northLine, *southLine, *westLine, *eastLine;
  GridPoint* point;
  GridModel* model;
  GridWidget* parent;
  QPointF newPos;

public:
  GridPointItem(GridPoint* point, GridModel* model, GridWidget* parent);
  virtual ~GridPointItem(void);

  virtual QRectF boundingRect() const;
  virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  virtual QPainterPath shape() const;

  void addLine(GridLine* line);
  GridPointItem* getNorth() const;
  GridPointItem* getSouth() const;
  GridPointItem* getWest() const;
  GridPointItem* getEast() const;

  void calculateForces();
  bool advance();

protected:
  virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
  virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);

private:
  void updateLines();
};

}

}

#endif /* GAPPUTILSCV_GRIDPOINTITEM_H_ */
