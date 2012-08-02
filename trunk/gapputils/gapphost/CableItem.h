#pragma once

#ifndef _CABLEITEM_H_
#define _CABLEITEM_H_

#include <qgraphicsitem.h>
#include <qpoint.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace capputils {
class ObservableClass;
}

namespace gapputils {

class ToolConnection;
class Workbench;

class CableItem : public QGraphicsItem
{
private:
  boost::weak_ptr<ToolConnection> input;
  boost::weak_ptr<ToolConnection> output;
  boost::shared_ptr<QPointF> dragPoint;

  QPointF sourcePoint;
  QPointF destPoint;

  Workbench* bench;

public:
  CableItem(Workbench* bench,
      boost::shared_ptr<ToolConnection> input = boost::shared_ptr<ToolConnection>(),
      boost::shared_ptr<ToolConnection> output = boost::shared_ptr<ToolConnection>());
  virtual ~CableItem(void);

  void adjust();
  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  void setDragPoint(QPointF point);
  void setInput(boost::shared_ptr<ToolConnection> input);
  void setOutput(boost::shared_ptr<ToolConnection> output);
  boost::shared_ptr<ToolConnection> getInput() const;
  boost::shared_ptr<ToolConnection> getOutput() const;
  bool needInput() const;
  bool needOutput() const;
  void endDrag();
};

}

#endif
