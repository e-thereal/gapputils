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
  std::vector<CableItem*> currentCables;
  bool modifiable;

public:
  Workbench(QWidget *parent = 0);
  virtual ~Workbench();

  void addToolItem(ToolItem* item);
  void addCableItem(CableItem* cable);
  void removeCableItem(CableItem* cable);
  void removeToolItem(ToolItem* item);

  void setSelectedItem(ToolItem* item);
  ToolItem* getSelectedItem() const;
  std::vector<CableItem*>& getCurrentCables();
  void notifyItemChange(ToolItem* item);

  void setModifiable(bool modifiable);

Q_SIGNALS:
  void createItemRequest(int x, int y, QString classname);
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
  void keyReleaseEvent(QKeyEvent *event);
  void dragEnterEvent(QDragEnterEvent *event);
  void dragMoveEvent(QDragMoveEvent* event);
  void dropEvent(QDropEvent *event);
  
  void wheelEvent(QWheelEvent *event);

  void scaleView(qreal scaleFactor);
};

}

#endif /* WORKBENCH_H_ */
