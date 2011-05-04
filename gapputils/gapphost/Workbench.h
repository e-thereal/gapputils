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
class CableItem;

class Workbench : public QGraphicsView {
  Q_OBJECT

private:
  ToolItem* selectedItem;
  CableItem* currentCable;

public:
  Workbench(QWidget *parent = 0);
  virtual ~Workbench();

  void addToolItem(ToolItem* item);
  void addCableItem(CableItem* cable);
  void removeCableItem(CableItem* cable);

  void setSelectedItem(ToolItem* item);
  ToolItem* getSelectedItem() const;
  CableItem* getCurrentCable() const;
  void notifyItemChange(ToolItem* item);

Q_SIGNALS:
  void itemSelected(ToolItem* item);
  void itemChanged(ToolItem* item);
  void itemDeleted(ToolItem* item);
  void cableCreated(CableItem* cable);
  void cableDeleted(CableItem* cable);

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void drawBackground(QPainter *painter, const QRectF &rect);
  void keyPressEvent(QKeyEvent *event);
  
  void wheelEvent(QWheelEvent *event);

  void scaleView(qreal scaleFactor);
};

}

#endif /* WORKBENCH_H_ */
