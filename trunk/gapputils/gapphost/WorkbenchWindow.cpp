/*
 * WorkbenchWindow.cpp
 *
 *  Created on: Aug 3, 2012
 *      Author: tombr
 */

#include "WorkbenchWindow.h"

#include "Workbench.h"
#include "DataModel.h"

#include "Node.h"
#include "Workflow.h"

#include "WorkflowItem.h"
#include "CableItem.h"

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/InterfaceAttribute.h>
#include <gapputils/WorkflowInterface.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Logbook.h>
#include <capputils/Xmlizer.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/LibraryLoader.h>

#include <qevent.h>

#include <qclipboard.h>
#include <qapplication.h>

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;

namespace gapputils {

using namespace attributes;
using namespace workflow;

namespace host {

WorkbenchWindow::WorkbenchWindow(boost::shared_ptr<workflow::Workflow> workflow, QWidget* parent)
: QMdiSubWindow(parent), workflow(workflow)
{
  Logbook& dlog = *workflow->getLogbook();

  workbench = new Workbench();
  setWidget(workbench);

  connect(workbench, SIGNAL(createItemRequest(int, int, QString)), this, SLOT(createModule(int, int, QString)));

  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *workflow->getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i)
    createItem(nodes[i]);

  std::vector<boost::shared_ptr<workflow::Edge> >& edges = *workflow->getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    if (!createCable(edges[i])) {
      workflow->removeEdge(edges[i]);
      --i;
      dlog() << "Edge has been removed from the model.";
    }
  }
}

WorkbenchWindow::~WorkbenchWindow() { }

const std::string& getPropertyLabel(capputils::reflection::IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

void WorkbenchWindow::createItem(boost::shared_ptr<workflow::Node> node) {
  ToolItem* item;
  assert(node->getModule());

  // Get the label
  std::string label = std::string("[") + node->getModule()->getClassName() + "]";
  std::vector<capputils::reflection::IClassProperty*>& properties = node->getModule()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      label = properties[i]->getStringValue(*node->getModule());
      break;
    }
  }

  boost::shared_ptr<workflow::Workflow> workflow = boost::dynamic_pointer_cast<workflow::Workflow>(node);
  if (workflow) {
    item = new WorkflowItem(label);
    connect((WorkflowItem*)item, SIGNAL(showWorkflowRequest(ToolItem*)), this, SLOT(showWorkflow(ToolItem*)));

    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>() && !prop->getAttribute<FromEnumerableAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>() && !prop->getAttribute<ToEnumerableAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
    }
  } else if (node->getModule()->getAttribute<InterfaceAttribute>()) {
    item = new ToolItem(label);
    connect(item, SIGNAL(showDialogRequested(ToolItem*)), this, SLOT(showModuleDialog(ToolItem*)));

    // Search for the value property and add it. Don't add any other stuff
    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      //if (prop->getName() != "Value")
      //  continue;
      if (prop->getAttribute<InputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
      //break;
    }
  } else {
    item = new ToolItem(label);
    connect(item, SIGNAL(showDialogRequested(ToolItem*)), this, SLOT(showModuleDialog(ToolItem*)));

    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
    }
  }

  item->setPos(node->getX(), node->getY());
  node->setToolItem(item);

  workbench->addToolItem(item);
  workbench->setCurrentItem(item);
}

bool WorkbenchWindow::createCable(boost::shared_ptr<workflow::Edge> edge) {
  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  Logbook& dlog = *workflow->getLogbook();

//  cout << "Connecting " << edge->getOutputNode() << "." << edge->getOutputProperty()
//       << " with " << edge->getInputNode() << "." << edge->getInputProperty() << "... " << flush;
//
  const std::string outputNodeUuid = edge->getOutputNode();
  const std::string inputNodeUuid = edge->getInputNode();

  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *workflow->getNodes();

  boost::shared_ptr<workflow::Node> outputNode, inputNode;

  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (nodes[i]->getUuid().compare(outputNodeUuid) == 0)
      outputNode = nodes[i];
    if (nodes[i]->getUuid().compare(inputNodeUuid) == 0)
      inputNode = nodes[i];
  }

  boost::shared_ptr<ToolConnection> outputConnection, inputConnection;
  if (outputNode && inputNode) {

    // TODO: try to find the correct propertyId. If the property is not a property
    //       of the module, go through the list of interface nodes and try to find
    //       the property there. PropertyNames of interface nodes are the node ID.
    unsigned outputPropertyId, inputPropertyId;
    if (workflow->getToolConnectionId(outputNode, edge->getOutputProperty(), outputPropertyId) &&
        workflow->getToolConnectionId(inputNode, edge->getInputProperty(), inputPropertyId))
    {
      outputConnection = outputNode->getToolItem()->getConnection(outputPropertyId, ToolConnection::Output);
      inputConnection = inputNode->getToolItem()->getConnection(inputPropertyId, ToolConnection::Input);

      if (outputConnection && inputConnection) {
        CableItem* cable = new CableItem(workbench, outputConnection, inputConnection);
        workbench->addCableItem(cable);
        edge->setCableItem(cable);

        edge->activate(outputNode, inputNode);
      }
    }
  } else {
    dlog(Severity::Warning) << "Can not find connections for edge '" << edge->getInputNode() << "' -> '" << edge->getOutputNode() << "'";
    return false;
  }
//  cout << "DONE!" << endl;
  return true;
}

void renewUuids(workflow::Workflow& workflow) {
  std::map<std::string, std::string> uuidMap;

  // Go through edges, nodes, global properties, global edges
  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *workflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    workflow::Node& node = *nodes[i];
    const std::string uuid = node.getUuid();
    if (uuidMap.find(uuid) == uuidMap.end())
      uuidMap[uuid] = workflow::Node::CreateUuid();
    node.setUuid(uuidMap[uuid]);

    // TODO: Not implemented unless the ID change is applied to all
    // occurances of an UUID in the workflow
//    Workflow* subworkflow = dynamic_cast<Workflow*>(&node);
//    if (subworkflow)
//      renewUuids(*subworkflow);
  }

  std::vector<boost::shared_ptr<workflow::Edge> >& edges = *workflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    workflow::Edge& edge = *edges[i];

    // Change IDs only if mapping is available. If no mapping is availabe,
    // the ID will likely not need a change (input or output node IDs)
    if (uuidMap.find(edge.getInputNode()) != uuidMap.end())
      edge.setInputNode(uuidMap[edge.getInputNode()]);
    if (uuidMap.find(edge.getOutputNode()) != uuidMap.end())
      edge.setOutputNode(uuidMap[edge.getOutputNode()]);
  }

  // TODO: replace UUIDs of other parts as well
}

void WorkbenchWindow::addNodesFromClipboard() {
  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  Logbook& dlog = *workflow->getLogbook();

  workflow::Workflow pasteWorkflow;
  const std::string clipboardText = QApplication::clipboard()->text().toUtf8().data();
  Xmlizer::FromXmlString(pasteWorkflow, clipboardText);

  // Unselect selected items
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems())
    item->setSelected(false);

  renewUuids(pasteWorkflow);
  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *pasteWorkflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    workflow->getNodes()->push_back(nodes[i]);
    workflow->resumeNode(nodes[i]);
    createItem(nodes[i]);
    nodes[i]->getToolItem()->setSelected(true);
  }
  nodes.clear(); // avoid double free memory

  std::vector<boost::shared_ptr<workflow::Edge> >& edges = *pasteWorkflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    workflow->getEdges()->push_back(edges[i]);
    if (!workflow->resumeEdge(edges[i]) || !createCable(edges[i])) {
      workflow->removeEdge(edges[i]);
      dlog() << "Edge has been removed from the model.";
    }
  }

  edges.clear(); // avoid double free memory
}

void WorkbenchWindow::closeEvent(QCloseEvent *event) {
  if (workflow.expired() || workflow.lock() != DataModel::getInstance().getMainWorkflow())
    event->accept();
  else
    event->ignore();
}

void addDependencies(boost::shared_ptr<Workflow> workflow, const std::string& classname) {
  // Update libraries
  std::string libName = LibraryLoader::getInstance().classDefinedIn(classname);
  if (libName.size()) {
    boost::shared_ptr<std::vector<std::string> > libraries = workflow->getLibraries();
    unsigned i = 0;
    for (; i < libraries->size(); ++i)
      if (libraries->at(i).compare(libName) == 0)
        break;
    if (i == libraries->size()) {
      libraries->push_back(libName);
      workflow->setLibraries(libraries);
    }
  }
}

void WorkbenchWindow::createModule(int x, int y, QString classname) {
  if (classname.count() == 0)
    return;

  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  std::string name = classname.toAscii().data();

  boost::shared_ptr<ReflectableClass> object = boost::shared_ptr<ReflectableClass>(ReflectableClassFactory::getInstance().newInstance(name));
  addDependencies(workflow, name);

  boost::shared_ptr<Node> node;
  if (boost::dynamic_pointer_cast<WorkflowInterface>(object)) {
    boost::shared_ptr<Workflow> workflow = boost::shared_ptr<Workflow>(new Workflow());
    workflow->setModule(object);
    addDependencies(workflow, name);
    workflow->resume();
    node = workflow;
  } else {
    node = boost::shared_ptr<Node>(new Node());
    node->setModule(object);
    node->resume();
  }
  node->setWorkflow(workflow);
  node->setX(x);
  node->setY(y);
  workflow->getNodes()->push_back(node);

  if (object->getAttribute<InterfaceAttribute>()) {
    workflow->addInterfaceNode(node);
  }

  createItem(node);
}

} /* namespace host */

} /* namespace gapputils */
