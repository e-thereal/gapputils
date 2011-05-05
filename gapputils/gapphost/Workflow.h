/*
 * Workflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef WORKFLOW_H_
#define WORKFLOW_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

#include <qobject.h>
#include <qwidget.h>
#include <vector>
#include <qtreeview.h>
#include <tinyxml.h>
#include <set>
#include "Edge.h"
#include "Node.h"

namespace gapputils {

class Workbench;
class ToolItem;
class CableItem;

namespace workflow {

class Workflow : public QObject, public Node, public capputils::ObservableClass
{
  Q_OBJECT

  InitReflectableClass(Workflow)

  Property(Libraries, std::vector<std::string>*)
  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)
  Property(InputsPosition, std::vector<int>)
  Property(OutputsPosition, std::vector<int>)

private:
  Workbench* workbench;
  QTreeView* propertyGrid;
  QWidget* widget;
  Node inputsNode, outputsNode;
  std::set<std::string> loadedLibraries;
  bool ownWidget;

public:
  Workflow();
  virtual ~Workflow();

  void newModule(const std::string& name);
  void newItem(Node* node);
  void newCable(Edge* edge);
  void resumeFromModel();
  /// The workflow loses ownership of the widget when calling this method
  QWidget* dispenseWidget();
  TiXmlElement* getXml(bool addEmptyModule = true) const;

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);

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
