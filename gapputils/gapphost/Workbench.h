/*
 * Workbench.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_WORKBENCH_H_
#define GAPPHOST_WORKBENCH_H_

#include <QGraphicsView>
#include <QGraphicsRectItem>

#include <set>

namespace gapputils {

class ToolItem;
class ToolConnection;
class CableItem;

class CompatibilityChecker {
public:
  virtual bool areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const = 0;
};

class Workbench : public QGraphicsView {
  Q_OBJECT

private:
  ToolItem* selectedItem;
  std::vector<CableItem*> currentCables;
  CompatibilityChecker* checker;
  bool modifiable;
  qreal viewScale;
  QGraphicsRectItem* shadowRect;
  std::set<QGraphicsItem*> dependentItems;

public:
  Workbench(QWidget *parent = 0);
  virtual ~Workbench();

  void addToolItem(ToolItem* item);
  void addCableItem(CableItem* cable);
  void setChecker(CompatibilityChecker* checker);

  // disconnects and deletes a cable and removes it from the list
  void removeCableItem(CableItem* cable);
  void removeToolItem(ToolItem* item);

  void setCurrentItem(ToolItem* item);
  ToolItem* getCurrentItem() const;
  std::vector<CableItem*>& getCurrentCables();
  void notifyItemChange(ToolItem* item);

  void setModifiable(bool modifiable);
  void scaleView(qreal scaleFactor);
  void setViewScale(qreal scale);
  qreal getViewScale();
  bool areCompatible(const ToolConnection* output, const ToolConnection* input) const;
  bool isDependent(QGraphicsItem* item);

Q_SIGNALS:
  void createItemRequest(int x, int y, QString classname);
  void currentItemSelected(ToolItem* item);
  void preItemDeleted(ToolItem* item);

  void itemChanged(ToolItem* item);

  void connectionCompleted(CableItem* cable);
  void connectionRemoved(CableItem* cable);
  void viewportChanged();

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
};

}

#endif /* WORKBENCH_H_ */
