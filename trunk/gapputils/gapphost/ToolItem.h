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

class ToolConnection {
public:
  enum Direction {Input, Output};

public:
  int x, y, width, height;
  QString label;
  Direction direction;
  ToolItem* parent;
  CableItem* cable;
  capputils::reflection::IClassProperty* property;
  int propertyId;

public:
  ToolConnection(const QString& label, Direction direction, ToolItem* parent, capputils::reflection::IClassProperty* property);
  virtual ~ToolConnection();

  void draw(QPainter* painter, bool showLabel = true) const;
  bool hit(int x, int y) const;
  void setPos(int x, int y);
  QPointF attachmentPos() const;
  void connect(CableItem* cable);
  void disconnect();
};

class ToolItem : public QGraphicsItem {
  friend class ToolConnection;

protected:
  workflow::Node* node;
  Workbench* bench;
  ModelHarmonizer harmonizer;
  int width, height, adjust, connectionDistance, inputsWidth, labelWidth, outputsWidth;
  std::vector<ToolConnection*> inputs;
  std::vector<ToolConnection*> outputs;
  QFont labelFont;

public:
  ToolItem(workflow::Node* node, Workbench *bench = 0);
  virtual ~ToolItem();

  void setWorkbench(Workbench* bench);
  workflow::Node* getNode() const;
  QAbstractItemModel* getModel() const;

  ToolConnection* hitConnection(int x, int y, ToolConnection::Direction direction) const;
  ToolConnection* getConnection(const std::string& propertyName, ToolConnection::Direction direction) const;
  void updateSize();
  void updateConnectionPositions();
  void drawConnections(QPainter* painter, bool showLabel = true);
  void drawBox(QPainter* painter);
  std::string getLabel() const;

  void mousePressEvent(QGraphicsSceneMouseEvent* event);
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
