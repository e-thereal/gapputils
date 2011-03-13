#pragma once

#ifndef _CABLEITEM_H_
#define _CABLEITEM_H_

#include <qgraphicsitem.h>
#include <qpoint.h>

namespace capputils {
class ObservableClass;
}

namespace gapputils {

class ToolConnection;
class Workbench;

class CableItem : public QGraphicsItem
{

  class ChangeEventHandler {
  private:
    CableItem* cableItem;

  public:
    ChangeEventHandler(CableItem* cableItem) : cableItem(cableItem) { }

    void operator()(capputils::ObservableClass*, int eventId);
    bool operator==(const ChangeEventHandler& handler) const;
  } changeEventHandler;

private:
  ToolConnection* input;
  ToolConnection* output;
  QPointF* dragPoint;

  QPointF sourcePoint;
  QPointF destPoint;

  Workbench* bench;

public:
  CableItem(Workbench* bench, ToolConnection* input = 0, ToolConnection* output = 0);
  virtual ~CableItem(void);

  void adjust();
  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  void setDragPoint(QPointF point);
  void setInput(ToolConnection* output);
  void setOutput(ToolConnection* output);
  ToolConnection* getInput() const;
  ToolConnection* getOutput() const;
  bool needInput() const;
  bool needOutput() const;
  void endDrag();
  void disconnectInput();
  void disconnectOutput();
};

}

#endif