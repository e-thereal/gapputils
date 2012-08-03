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

public:
  QString label;
  ToolConnection::Direction direction;
  ToolItem* parent;
  int id;
  std::vector<boost::shared_ptr<ToolConnection> > connections;
  int x, y;
  bool expanded;

public:
  MultiConnection(const QString& label, ToolConnection::Direction direction, ToolItem* parent,
      int id);
  virtual ~MultiConnection();

  bool hits(std::vector<boost::shared_ptr<ToolConnection> >& connections, int x, int y) const;
  boost::shared_ptr<ToolConnection> getLastConnection();
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
  std::vector<boost::shared_ptr<ToolConnection> > inputs;
  std::vector<boost::shared_ptr<MultiConnection> > outputs;
  QFont labelFont;
  double progress;

public:
  ToolItem(const std::string& label, Workbench* bench = 0);
  virtual ~ToolItem();

  void setWorkbench(Workbench* bench);

  boost::shared_ptr<ToolConnection> hitConnection(int x, int y, ToolConnection::Direction direction) const;

  /// Adds the connections to the connections vector
  bool hitConnections(std::vector<boost::shared_ptr<ToolConnection> >& connections, int x, int y, ToolConnection::Direction direction) const;

  boost::shared_ptr<ToolConnection> getConnection(int id, ToolConnection::Direction direction) const;
  std::vector<boost::shared_ptr<ToolConnection> >& getInputs();
  void getOutputs(std::vector<boost::shared_ptr<ToolConnection> >& connections);
  void updateSize();

  void updateConnectionPositions();
  void addConnection(const QString& label, int id, ToolConnection::Direction direction);
  void deleteConnection(int id, ToolConnection::Direction direction);
  void drawConnections(QPainter* painter, bool showLabel = true);
  void drawBox(QPainter* painter);
  virtual std::string getLabel() const;
  void setLabel(const std::string& label);

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
