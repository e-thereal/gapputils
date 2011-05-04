/*
 * Workflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef WORKFLOW_H_
#define WORKFLOW_H_

#include <ReflectableClass.h>

#include <qobject.h>
#include <qwidget.h>
#include <vector>
#include <qtreeview.h>
#include "Edge.h"
#include "Node.h"

namespace gapputils {

class Workbench;
class ToolItem;
class CableItem;

namespace workflow {

class Workflow : public QObject, public capputils::reflection::ReflectableClass
{
  Q_OBJECT

  InitReflectableClass(Workflow)

  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)
  Property(InputsPosition, std::vector<int>)
  Property(OutputsPosition, std::vector<int>)

  Workbench* workbench;
  QTreeView* propertyGrid;
  QWidget* widget;
  Node inputsNode, outputsNode;

public:
  Workflow();
  virtual ~Workflow();

  void newModule(const std::string& name);
  void newItem(Node* node);
  void newCable(Edge* edge);
  void resumeFromModel();
  QWidget* getWidget();

private Q_SLOTS:
  void itemSelected(ToolItem* item);
  void itemChangedHandler(ToolItem* item);
  void deleteItem(ToolItem* item);
  void createEdge(CableItem* cable);
  void deleteEdge(CableItem* cable);
};

}

}

#endif /* WORKFLOW_H_ */
