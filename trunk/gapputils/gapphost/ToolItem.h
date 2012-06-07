/*
 * ToolItem.h
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#ifndef TOOLITEM_H_
#define TOOLITEM_H_

#include <qgraphicsitem.h>
#include <capputils/ReflectableClass.h>
#include <qabstractitemmodel.h>
#include <capputils/ObservableClass.h>
#include <vector>

#include "ModelHarmonizer.h"
#include "Node.h"

namespace gapputils {

class Workbench;
class ToolItem;
class CableItem;
class MultiConnection;

class ToolConnection {
public:
  enum Direction {Input, Output};

public:
  int x, y, width, height;
  QString label;
  Direction direction;
  ToolItem* parent;
  MultiConnection* multi;
  CableItem* cable;
  int id;                 ///< PropertyId

public:
  ToolConnection(const QString& label, Direction direction, ToolItem* parent,
      int id, MultiConnection* multi = 0);
  virtual ~ToolConnection();

  void draw(QPainter* painter, bool showLabel = true) const;
  bool hit(int x, int y) const;
  void setPos(int x, int y);
  QPointF attachmentPos() const;
  void connect(CableItem* cable);
};

class MultiConnection {
  friend class ToolItem;

private:
  QString label;
  ToolConnection::Direction direction;
  ToolItem* parent;
  int id;
  std::vector<ToolConnection*> connections;
  int x, y;
  bool expanded;

public:
  MultiConnection(const QString& label, ToolConnection::Direction direction, ToolItem* parent,
      int id);
  virtual ~MultiConnection();

  bool hits(std::vector<ToolConnection*>& connections, int x, int y) const;
  ToolConnection* getLastConnection();
  void adjust();
  QString getLabel() const;
  void setPos(int x, int y);
  void updateConnections();
  void updateConnectionPositions();
  void draw(QPainter* painter, bool showLabel = true) const;
  bool clickEvent(int x, int y);
  int getHeight() const;
};

class ToolItem : public QObject, public QGraphicsItem {
  Q_OBJECT

  friend class ToolConnection;
  friend class MultiConnection;

public:
  enum ProgressStates {Neutral = -1, InProgress = -2};

protected:
  std::string label;
  Workbench* bench;
  int width, height, adjust, connectionDistance, inputsWidth, labelWidth, outputsWidth;
  std::vector<ToolConnection*> inputs;
  std::vector<MultiConnection*> outputs;
  QFont labelFont;
  bool deletable;
  double progress;

public:
  ToolItem(const std::string& label, Workbench *bench = 0);
  virtual ~ToolItem();

  void setWorkbench(Workbench* bench);

  ToolConnection* hitConnection(int x, int y, ToolConnection::Direction direction) const;

  /// Adds the connections to the connections vector
  bool hitConnections(std::vector<ToolConnection*>& connections, int x, int y, ToolConnection::Direction direction) const;

  ToolConnection* getConnection(int id, ToolConnection::Direction direction) const;
  std::vector<ToolConnection*>& getInputs();
  void getOutputs(std::vector<ToolConnection*>& connections);
  void updateSize();

  void updateConnectionPositions();
  void addConnection(const QString& label, int id, ToolConnection::Direction direction);
  void drawConnections(QPainter* painter, bool showLabel = true);
  void drawBox(QPainter* painter);
  virtual std::string getLabel() const;
  void setLabel(const std::string& label);
  virtual bool isDeletable() const;
  void setDeletable(bool deletable);

  void updateCables();

  /// progess is in %
  void setProgress(double progress);

  virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
  virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
  virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);
  QVariant itemChange(GraphicsItemChange change, const QVariant &value);

  QRectF boundingRect() const;
  QPainterPath shape() const;
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  bool isCurrentItem() const;

Q_SIGNALS:
  void showDialogRequested(ToolItem* item);
};

}

#endif /* TOOLITEM_H_ */
