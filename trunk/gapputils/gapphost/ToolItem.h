/*
 * ToolItem.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef TOOLITEM_H_
#define TOOLITEM_H_

#include <qgraphicsitem.h>
#include <ReflectableClass.h>
#include <qabstractitemmodel.h>

#include "ModelHarmonizer.h"

namespace gapputils {

class Workbench;

class ToolItem : public QGraphicsItem {
private:
  capputils::reflection::ReflectableClass* object;
  Workbench* bench;
  ModelHarmonizer harmonizer;

public:
  ToolItem(capputils::reflection::ReflectableClass* object, Workbench *bench = 0);
  virtual ~ToolItem();

  void setWorkbench(Workbench* bench);
  capputils::reflection::ReflectableClass* getObject() const;
  QAbstractItemModel* getModel() const;

  void mousePressEvent(QGraphicsSceneMouseEvent* event);

  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

}

#endif /* TOOLITEM_H_ */
