#pragma once

#ifndef _CABLEITEM_H_
#define _CABLEITEM_H_

#include <qgraphicsitem.h>
#include <qpoint.h>

namespace gapputils {

class ToolConnection;

class CableItem : public QGraphicsItem
{
private:
  ToolConnection* input;
  ToolConnection* output;
  QPointF* dragPoint;

  QPointF sourcePoint;
  QPointF destPoint;

public:
  CableItem(ToolConnection* input = 0, ToolConnection* output = 0);
  virtual ~CableItem(void);

  void adjust();
  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  void setDragPoint(QPointF point);
  void setInput(ToolConnection* output);
  void setOutput(ToolConnection* output);
  bool needInput() const;
  bool needOutput() const;
  void endDrag();
  void disconnectInput();
  void disconnectOutput();
};

}

#endif