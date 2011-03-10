/*
 * Workbench.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef WORKBENCH_H_
#define WORKBENCH_H_

#include <QGraphicsView>

namespace gapputils {

class ToolItem;

class Workbench : public QGraphicsView {
  Q_OBJECT

private:
  ToolItem* selectedItem;

public:
  Workbench(QWidget *parent = 0);
  virtual ~Workbench();

  void addToolItem(ToolItem* item);

  void setSelectedItem(ToolItem* item);
  ToolItem* getSelectedItem();

Q_SIGNALS:
  void itemSelected(ToolItem* item);

protected:
  void drawBackground(QPainter *painter, const QRectF &rect);
  void wheelEvent(QWheelEvent *event);

  void scaleView(qreal scaleFactor);
};

}

#endif /* WORKBENCH_H_ */
