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
#include <ObservableClass.h>
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
  capputils::reflection::IClassProperty* property;
  int propertyId;
  bool deleting;

public:
  ToolConnection(const QString& label, Direction direction, ToolItem* parent,
      capputils::reflection::IClassProperty* property, MultiConnection* multi = 0);
  virtual ~ToolConnection();

  void draw(QPainter* painter, bool showLabel = true) const;
  bool hit(int x, int y) const;
  void setPos(int x, int y);
  QPointF attachmentPos() const;
  void connect(CableItem* cable);
  void disconnect();
};

class MultiConnection {
private:
  QString label;
  ToolConnection::Direction direction;
  ToolItem* parent;
  capputils::reflection::IClassProperty* property;
  std::vector<ToolConnection*> connections;
  int x, y;
  bool expanded;
  bool deleting;

public:
  MultiConnection(const QString& label, ToolConnection::Direction direction, ToolItem* parent,
      capputils::reflection::IClassProperty* property);
  virtual ~MultiConnection();

  ToolConnection* hit(int x, int y);
  ToolConnection* getLastConnection();
  capputils::reflection::IClassProperty* getProperty() const;
  void adjust();
  QString getLabel() const;
  void setPos(int x, int y);
  void updateConnections();
  void updateConnectionPositions();
  void draw(QPainter* painter, bool showLabel = true) const;
  bool clickEvent(int x, int y);
  int getHeight() const;
};

class ToolItem : public QGraphicsItem {
  friend class ToolConnection;
  friend class MultiConnection;

protected:
  workflow::Node* node;
  Workbench* bench;
  ModelHarmonizer harmonizer;
  int width, height, adjust, connectionDistance, inputsWidth, labelWidth, outputsWidth;
  std::vector<ToolConnection*> inputs;
  std::vector<MultiConnection*> outputs;
  QFont labelFont;
  bool deletable;
  int progress;
  bool deleting;

public:
  ToolItem(workflow::Node* node, Workbench *bench = 0);
  virtual ~ToolItem();

  void setWorkbench(Workbench* bench);
  workflow::Node* getNode() const;
  void setNode(workflow::Node* node);
  QAbstractItemModel* getModel() const;

  ToolConnection* hitConnection(int x, int y, ToolConnection::Direction direction) const;
  ToolConnection* getConnection(const std::string& propertyName, ToolConnection::Direction direction) const;
  std::vector<ToolConnection*>& getInputs();
  void updateSize();

  void updateConnectionPositions();
  void drawConnections(QPainter* painter, bool showLabel = true);
  void drawBox(QPainter* painter);
  virtual std::string getLabel() const;
  virtual void updateConnections();
  virtual bool isDeletable() const;

  void updateCables();

  /// progess is in %
  void setProgress(int progress);

  void mousePressEvent(QGraphicsSceneMouseEvent* event);
  void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
  QVariant itemChange(GraphicsItemChange change, const QVariant &value);
  
  QRectF boundingRect() const;
  QPainterPath shape() const;
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  bool isSelected() const;

protected:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif /* TOOLITEM_H_ */
