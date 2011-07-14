/*
 * RectangleItem.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RECTANGLEITEM_H_
#define GAPPUTILSCV_RECTANGLEITEM_H_

#include <QGraphicsItem>

#include <boost/shared_ptr.hpp>

#include <capputils/ObservableClass.h>
#include <capputils/EventHandler.h>

namespace gapputils {

namespace cv {

class RectangleModel;
class RectangleWidget;

class RectangleItem : public QGraphicsItem {
private:
  boost::shared_ptr<RectangleModel> model;
  RectangleWidget* parent;
  capputils::EventHandler<RectangleItem> handler;

public:
  RectangleItem(boost::shared_ptr<RectangleModel> model, RectangleWidget* parent);
  virtual ~RectangleItem();

  virtual QRectF boundingRect() const;
  virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  virtual QPainterPath shape() const;

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_RECTANGLEITEM_H_ */
