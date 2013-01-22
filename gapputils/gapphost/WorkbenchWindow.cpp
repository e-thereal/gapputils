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
#include "DataModel.h"

#include "WorkflowItem.h"
#include "CableItem.h"
#include "MainWindow.h"

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/InterfaceAttribute.h>
#include <gapputils/WorkflowInterface.h>
#include <gapputils/WorkflowElement.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Logbook.h>
#include <capputils/Xmlizer.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/LibraryLoader.h>
#include <capputils/ObservableClass.h>

#include <qevent.h>
#include <qmdiarea.h>
#include <qlabel.h>
#include <qclipboard.h>
#include <qapplication.h>
#include <qmessagebox.h>

#include <iostream>
#include <iomanip>
#include <cmath>

#include "WorkflowUpdater.h"
#include "DataModel.h"
#include "MakeGlobalDialog.h"
#include "WorkflowSnippets.h"

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;

namespace gapputils {

using namespace attributes;
using namespace workflow;

namespace host {

void addDependencies(boost::shared_ptr<Workflow> workflow, const std::string& classname) {
  // Update libraries
  std::string libName = LibraryLoader::getInstance().classDefinedIn(classname);

  std::string path = DataModel::getInstance().getLibraryPath();

  if (path.size() && libName.size() > path.size()) {
    if (libName.compare(0, path.size(), path) == 0) {
      if (path.size() > 1 && path[path.size() - 1] == '/')
        libName = libName.substr(path.size());
      else
        libName = libName.substr(path.size() + 1);
    }
  }

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

WorkbenchWindow::WorkbenchWindow(boost::shared_ptr<workflow::Workflow> workflow, QWidget* parent)
: QMdiSubWindow(parent), workflow(workflow), workflowUpdater(new WorkflowUpdater()),
  handler(this, &WorkbenchWindow::changedHandler),
  modelEventHandler(this, &WorkbenchWindow::handleModelEvents),
  closable(true)
{
  setAttribute(Qt::WA_DeleteOnClose);

  Logbook& dlog = *workflow->getLogbook();

  setWidget(workbench = new Workbench());
  workbench->setChecker(workflow.get());

  connect(workbench, SIGNAL(createItemRequest(int, int, QString)), this, SLOT(createModule(int, int, QString)));
  connect(workbench, SIGNAL(connectionCompleted(CableItem*)), this, SLOT(createEdge(CableItem*)));
  connect(workbench, SIGNAL(connectionRemoved(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  connect(workbench, SIGNAL(preItemDeleted(ToolItem*)), this, SLOT(deleteModule(ToolItem*)));

  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(currentItemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
  connect(workbench, SIGNAL(viewportChanged()), this, SLOT(handleViewportChanged()));

  connect(workflowUpdater.get(), SIGNAL(updateFinished()), this, SLOT(workflowUpdateFinished()));
  connect(workflowUpdater.get(), SIGNAL(progressed(boost::shared_ptr<workflow::Node>, double)), this, SLOT(showProgress(boost::shared_ptr<workflow::Node>, double)));

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

  boost::shared_ptr<WorkflowInterface> winterface;
  if (DataModel::getInstance().getMainWorkflow() == workflow) {
    setWindowTitle("Main Workflow");
  } else if ((winterface = boost::dynamic_pointer_cast<WorkflowInterface>(workflow->getModule()))) {
    setWindowTitle(winterface->getLabel().c_str());
  } else {
    setWindowTitle("Unknown Title");
  }

  capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(workflow->getModule().get());
  if (observable) {
    observable->Changed.connect(handler);
  }

  DataModel::getInstance().Changed.connect(modelEventHandler);
}

WorkbenchWindow::~WorkbenchWindow() {
  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  if (workflow) {
    std::vector<boost::shared_ptr<Node> >& nodes = *workflow->getNodes();
    for (unsigned i = 0; i < nodes.size(); ++i)
      nodes[i]->setToolItem(0);

    std::vector<boost::shared_ptr<Edge> >& edges = *workflow->getEdges();
    for (unsigned i = 0; i < edges.size(); ++i)
      edges[i]->setCableItem(0);

    capputils::ObservableClass* observable = dynamic_cast<capputils::ObservableClass*>(workflow->getModule().get());
    if (observable) {
      observable->Changed.disconnect(handler);
    }
  }
  DataModel::getInstance().Changed.disconnect(modelEventHandler);
}

const std::string& getPropertyLabel(capputils::reflection::IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

boost::shared_ptr<workflow::Workflow> WorkbenchWindow::getWorkflow() const {
  return workflow.lock();
}

void WorkbenchWindow::setClosable(bool closable) {
  this->closable = closable;
}

bool WorkbenchWindow::getClosable() const {
  return closable;
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

    // Add tool connections for properties
    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>() && !prop->getAttribute<FromEnumerableAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>() && !prop->getAttribute<ToEnumerableAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Output);
    }

    // Add tool connections for interface nodes
    std::vector<boost::weak_ptr<Node> >& nodes = workflow->getInterfaceNodes();
    for (unsigned i = 0; i < nodes.size(); ++i) {
      boost::shared_ptr<Node> node = nodes[i].lock();
      boost::shared_ptr<ReflectableClass> object = node->getModule();
      assert(object);

      IClassProperty* prop = object->findProperty("Value");
      if (!prop)
        return;

      if (prop->getAttribute<InputAttribute>())
        item->addConnection(QString(object->getProperty("Label").c_str()), node->getUuid(), ToolConnection::Output);
      
      if (prop->getAttribute<OutputAttribute>()) {
        item->addConnection(QString(object->getProperty("Label").c_str()), node->getUuid(), ToolConnection::Input);
      }
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
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Output);
      //break;
    }
  } else {
    item = new ToolItem(label);
    connect(item, SIGNAL(showDialogRequested(ToolItem*)), this, SLOT(showModuleDialog(ToolItem*)));

    for (unsigned i = 0; i < properties.size(); ++i) {
      capputils::reflection::IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), prop->getName(), ToolConnection::Output);
    }
  }

  item->setPos(node->getX(), node->getY());
  node->setToolItem(item);

  workbench->addToolItem(item);
  workbench->setCurrentItem(item);
}

bool WorkbenchWindow::createCable(boost::shared_ptr<workflow::Edge> edge) {

  // Remark: The edge is assumed to be already active

  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  Logbook& dlog = *workflow->getLogbook();

//  cout << "Connecting " << edge->getOutputNode() << "." << edge->getOutputProperty()
//       << " with " << edge->getInputNode() << "." << edge->getInputProperty() << "... " << flush;
//
  const std::string outputNodeUuid = edge->getOutputNode();
  const std::string inputNodeUuid = edge->getInputNode();

  boost::shared_ptr<workflow::Node> outputNode = workflow->getNode(edge->getOutputNode());
  boost::shared_ptr<workflow::Node> inputNode = workflow->getNode(edge->getInputNode());

  if (outputNode && inputNode) {
    boost::shared_ptr<ToolConnection> outputConnection = outputNode->getToolItem()->getConnection(edge->getOutputProperty(), ToolConnection::Output);
    boost::shared_ptr<ToolConnection> inputConnection = inputNode->getToolItem()->getConnection(edge->getInputProperty(), ToolConnection::Input);

    if (outputConnection && inputConnection) {
      CableItem* cable = new CableItem(workbench, outputConnection, inputConnection);
      workbench->addCableItem(cable);
      edge->setCableItem(cable);
      return true;
    }
  }
  dlog(Severity::Warning) << "Can not find connections for edge '" << edge->getInputNode() << "' -> '" << edge->getOutputNode() << "'";
  return false;
}

boost::shared_ptr<workflow::Workflow> WorkbenchWindow::copySelectedNodes() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  boost::shared_ptr<Workflow> copyWorkflow(new Workflow());
  std::set<std::string> copied;

  // Temporarily add nodes to the node list for the xmlization.
  // Nodes have to be removed afterwards in order to avoid a double free memory (is this really true? I don't think so!)
  boost::shared_ptr<std::vector<boost::shared_ptr<Node> > > nodes = copyWorkflow->getNodes();
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems()) {
    ToolItem* toolItem = dynamic_cast<ToolItem*>(item);
    if (toolItem) {
      boost::shared_ptr<Node> node = workflow->getNode(toolItem);
      nodes->push_back(node);
      copied.insert(node->getUuid());
    }
  }

  // Add all edges to the workflow where both end nodes are about to be copied
  boost::shared_ptr<std::vector<boost::shared_ptr<Edge> > > edges = copyWorkflow->getEdges();
  for (unsigned i = 0; i < workflow->getEdges()->size(); ++i) {
    boost::shared_ptr<Edge> edge = workflow->getEdges()->at(i);
    if (copied.find(edge->getInputNode()) != copied.end() &&
        copied.find(edge->getOutputNode()) != copied.end())
    {
      edges->push_back(edge);
    }
  }

  return copyWorkflow;
}

void renewUuids(workflow::Workflow& workflow, std::map<std::string, std::string>& uuidMap) {

  // Go through edges, nodes, global properties, global edges
  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *workflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    workflow::Node& node = *nodes[i];
    const std::string uuid = node.getUuid();
    if (uuidMap.find(uuid) == uuidMap.end())
      uuidMap[uuid] = workflow::Node::CreateUuid();
    node.setUuid(uuidMap[uuid]);

    Workflow* subworkflow = dynamic_cast<Workflow*>(&node);
    if (subworkflow)
      renewUuids(*subworkflow, uuidMap);
  }

  std::vector<boost::shared_ptr<workflow::Edge> >& edges = *workflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    workflow::Edge& edge = *edges[i];

    // Change IDs only if mapping is available. If no mapping is availabe,
    // the ID will likely not need a change
    if (uuidMap.find(edge.getInputNode()) != uuidMap.end())
      edge.setInputNode(uuidMap[edge.getInputNode()]);
    if (uuidMap.find(edge.getOutputNode()) != uuidMap.end())
      edge.setOutputNode(uuidMap[edge.getOutputNode()]);

    if (uuidMap.find(edge.getInputProperty()) != uuidMap.end())
      edge.setInputProperty(uuidMap[edge.getInputProperty()]);
    if (uuidMap.find(edge.getOutputProperty()) != uuidMap.end())
      edge.setOutputProperty(uuidMap[edge.getOutputProperty()]);
  }

  std::vector<boost::shared_ptr<workflow::GlobalProperty> >& gprops = *workflow.getGlobalProperties();
  for (unsigned i = 0; i < gprops.size(); ++i) {
    workflow::GlobalProperty& gprop = *gprops[i];

    if (uuidMap.find(gprop.getModuleUuid()) != uuidMap.end())
      gprop.setModuleUuid(uuidMap[gprop.getModuleUuid()]);
    if (uuidMap.find(gprop.getPropertyId()) != uuidMap.end())
      gprop.setPropertyId(uuidMap[gprop.getPropertyId()]);
  }

  std::vector<boost::shared_ptr<workflow::GlobalEdge> >& gedges = *workflow.getGlobalEdges();
  for (unsigned i = 0; i < gedges.size(); ++i) {
    workflow::GlobalEdge& edge = *gedges[i];

    // Change IDs only if mapping is available. If no mapping is availabe,
    // the ID will likely not need a change
    if (uuidMap.find(edge.getInputNode()) != uuidMap.end())
      edge.setInputNode(uuidMap[edge.getInputNode()]);
    if (uuidMap.find(edge.getOutputNode()) != uuidMap.end())
      edge.setOutputNode(uuidMap[edge.getOutputNode()]);

    if (uuidMap.find(edge.getInputProperty()) != uuidMap.end())
      edge.setInputProperty(uuidMap[edge.getInputProperty()]);
    if (uuidMap.find(edge.getOutputProperty()) != uuidMap.end())
      edge.setOutputProperty(uuidMap[edge.getOutputProperty()]);
  }
}

void renewUuids(workflow::Workflow& workflow) {
  std::map<std::string, std::string> uuidMap;
  renewUuids(workflow, uuidMap);
}

void WorkbenchWindow::addNodes(workflow::Workflow& pasteWorkflow) {
  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  Logbook& dlog = *workflow->getLogbook();

  // Unselect selected items
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems())
    item->setSelected(false);

  renewUuids(pasteWorkflow);
  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *pasteWorkflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (nodes[i]->getModule())
      addDependencies(workflow, nodes[i]->getModule()->getClassName());
    workflow->getNodes()->push_back(nodes[i]);
    workflow->resumeNode(nodes[i]);
    createItem(nodes[i]);
    nodes[i]->getToolItem()->setSelected(true);
  }

  std::vector<boost::shared_ptr<workflow::Edge> >& edges = *pasteWorkflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    workflow->getEdges()->push_back(edges[i]);
    if (!workflow->resumeEdge(edges[i]) || !createCable(edges[i])) {
      workflow->removeEdge(edges[i]);
      dlog() << "Edge has been removed from the model.";
    }
  }
}

void WorkbenchWindow::copySelectedNodesToClipboard() {
  boost::shared_ptr<Workflow> copyWorkflow = copySelectedNodes();

  std::stringstream xmlStream;
  Xmlizer::ToXml(xmlStream, *copyWorkflow);

  QApplication::clipboard()->setText(xmlStream.str().c_str());
}

void WorkbenchWindow::addNodesFromClipboard() {
  workflow::Workflow pasteWorkflow;
  const std::string clipboardText = QApplication::clipboard()->text().toUtf8().data();
  Xmlizer::FromXmlString(pasteWorkflow, clipboardText);
  addNodes(pasteWorkflow);
}

void WorkbenchWindow::createSnippet() {
  boost::shared_ptr<Workflow> copyWorkflow = copySelectedNodes();

  MakeGlobalDialog dialog(this);
  if (dialog.exec() == QDialog::Accepted) {
    std::string snippetName = dialog.getText().toAscii().data();
    if (snippetName.size()) {
      Xmlizer::ToXml(DataModel::getInstance().getSnippetsPath() + "/" + snippetName + ".xml", *copyWorkflow);
      WorkflowSnippets::GetInstance().update();
    } else {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for a workflow snippet!");
    }
  }
}

void WorkbenchWindow::addNodesFromSnippet(int x, int y, const std::string& filename) {
  workflow::Workflow pasteWorkflow;
  Xmlizer::FromXml(pasteWorkflow, filename);

  // Calculate top left
  std::vector<boost::shared_ptr<workflow::Node> >& nodes = *pasteWorkflow.getNodes();
  int left = 0, top = 0;
  if (nodes.size())
    left = nodes[0]->getX(), top = nodes[0]->getY();
  for (size_t i = 1; i < nodes.size(); ++i)
    left = std::min(left, nodes[i]->getX()), top = std::min(top, nodes[i]->getY());

  // Move top left to drop position
  for (size_t i = 0; i < nodes.size(); ++i)
    nodes[i]->setX(nodes[i]->getX() + x - left), nodes[i]->setY(nodes[i]->getY() + y - top);

  addNodes(pasteWorkflow);
}

void WorkbenchWindow::closeEvent(QCloseEvent *event) {
  if (closable) {
    event->accept();
  } else {
    event->ignore();
  }
  QMdiSubWindow::closeEvent(event);
}

void WorkbenchWindow::setUiEnabled(bool enabled) {
  workbench->setModifiable(enabled);
}

void WorkbenchWindow::resumeViewport() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  std::vector<double> pos = workflow->getViewportPosition();
  workbench->setViewScale(workflow->getViewportScale());
  workbench->centerOn(pos[0], pos[1]);
  handleViewportChanged();
}

bool WorkbenchWindow::trySelectNode(const std::string& uuid) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  boost::shared_ptr<Node> node = workflow->getNode(uuid);
  if (node && node->getToolItem()) {
    workbench->setExclusivelySelected(node->getToolItem());
    workbench->setCurrentItem(node->getToolItem());
    return true;
  }

  return false;
}

boost::shared_ptr<workflow::Node> WorkbenchWindow::getCurrentNode() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  return workflow->getNode(workbench->getCurrentItem());
}

void WorkbenchWindow::updateCurrentModule() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  boost::shared_ptr<Node> node = workflow->getNode(workbench->getCurrentItem());
  if (!node)
    return;

  // update checksums before updating the workflow
  workflowUpdater->update(node);
}

void WorkbenchWindow::updateInputs() {
  // TODO: implement update inputs
}

void WorkbenchWindow::updateOutputs() {
  workflowUpdater->update(workflow.lock());
}

void WorkbenchWindow::updateNodeByLabel(const std::string& label) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();
  Logbook& dlog = *workflow->getLogbook();

  boost::shared_ptr<Node> node = workflow->getNodeByLabel(label);
  if(!node) {
    dlog(Severity::Error) << "Could not find node with the label '" << label << "'. Won't update workflow.";
    return;
  }

  workflowUpdater->update(node);
}

void WorkbenchWindow::abortUpdate() {
  workflowUpdater->abort();
}

void WorkbenchWindow::changedHandler(capputils::ObservableClass* sender, int eventId) {
  workflow::WorkflowInterface* interface = dynamic_cast<workflow::WorkflowInterface*>(sender);
  
  if (interface && eventId == workflow::WorkflowInterface::LabelId) {
    setWindowTitle(interface->getLabel().c_str());
  }
}

void WorkbenchWindow::handleModelEvents(capputils::ObservableClass* sender, int eventId) {
  if (eventId == DataModel::WorkflowMapId && workflow.expired()) {
    setClosable(true);
    close();
  }
}

/*** SLOTS ***/

boost::shared_ptr<workflow::Node> WorkbenchWindow::createModule(int x, int y, QString classname) {
  boost::shared_ptr<Node> node;

  Logbook& dlog = *workflow.lock()->getLogbook();

  if (classname.endsWith(".xml")) {
    addNodesFromSnippet(x, y, classname.toAscii().data());
    return boost::shared_ptr<workflow::Node>();
  }

  dlog() << "Creating module.";
  if (classname.count() == 0)
    return node;

  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();
  std::string name = classname.toAscii().data();

  boost::shared_ptr<ReflectableClass> object = boost::shared_ptr<ReflectableClass>(ReflectableClassFactory::getInstance().newInstance(name));
  addDependencies(workflow, name);

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
  return node;
}

void WorkbenchWindow::createEdge(CableItem* cable) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  boost::shared_ptr<Node> outputNode = workflow->getNode(cable->getInput()->parent);
  boost::shared_ptr<Node> inputNode = workflow->getNode(cable->getOutput()->parent);

  // Sanity check. Should never fail
  assert(outputNode && outputNode->getModule() && inputNode && inputNode->getModule());

  boost::shared_ptr<Edge> edge(new Edge());

  edge->setOutputNode(outputNode->getUuid());
  edge->setOutputProperty(cable->getInput()->id);
  edge->setInputNode(inputNode->getUuid());
  edge->setInputProperty(cable->getOutput()->id);

  edge->setCableItem(cable);
  if (!edge->activate(outputNode, inputNode)) {
    workbench->removeCableItem(cable);
  } else {
    workflow->getEdges()->push_back(edge);
  }
}

void WorkbenchWindow::deleteEdge(CableItem* cable) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();
  workflow->removeEdge(workflow->getEdge(cable));
}

void WorkbenchWindow::deleteModule(ToolItem* item) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();
  workflow->removeNode(workflow->getNode(item));
}

void WorkbenchWindow::itemChangedHandler(ToolItem* item) {
  boost::shared_ptr<Node> node = workflow.lock()->getNode(item);
  if (node) {
    node->setX(item->x());
    node->setY(item->y());
  }
}

void WorkbenchWindow::itemSelected(ToolItem* item) {
  DataModel& model = DataModel::getInstance();
  if (item)
    model.getMainWindow()->handleCurrentNodeChanged(workflow.lock()->getNode(item));
  else
    model.getMainWindow()->handleCurrentNodeChanged(workflow.lock());
}

void WorkbenchWindow::showModuleDialog(ToolItem* item) {
  boost::shared_ptr<workflow::Workflow> workflow = this->workflow.lock();

  boost::shared_ptr<Node> node = workflow->getNode(item);
  boost::shared_ptr<WorkflowElement> element = boost::dynamic_pointer_cast<WorkflowElement>(node->getModule());
  if (element)
    element->show();
}

void WorkbenchWindow::showWorkflow(ToolItem* item) {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();
  DataModel& model = DataModel::getInstance();

  boost::shared_ptr<Workflow> subworkflow = boost::dynamic_pointer_cast<Workflow>(workflow->getNode(item));
  if (subworkflow) {
    model.getMainWindow()->showWorkflow(subworkflow);
  }
}

void WorkbenchWindow::handleViewportChanged() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  QPointF cnt = workbench->mapToScene(workbench->viewport()->rect().center());

  workflow->setViewportScale(workbench->getViewScale());
  std::vector<double> position;
  position.push_back(cnt.x());
  position.push_back(cnt.y());
  workflow->setViewportPosition(position);
}

std::string formatTime(int seconds) {
  int minutes = 0, hours = 0, days = 0;
  minutes = seconds / 60;
  seconds -= 60 * minutes;

  hours = minutes / 60;
  minutes -= 60 * hours;

  days = hours / 24;
  hours -= 24 * days;

  std::stringstream out;

  out << std::setfill('0');
  if (days) {
    out << days << "d ";
    out << std::setw(2) << hours << "h ";
    out << std::setw(2) << minutes << "min ";
    out << std::setw(2) << seconds << "s";
  } else if (hours) {
    out << hours << "h ";
    out << std::setw(2) << minutes << "min ";
    out << std::setw(2) << seconds << "s";
  } else if (minutes) {
    out << minutes << "min ";
    out << std::setw(2) << seconds << "s";
  } else {
    out << seconds << "s";
  }

  // maximum length is 19.
  return out.str() + std::string(25 - out.str().size(), ' ');
}

void WorkbenchWindow::showProgress(boost::shared_ptr<Node> node, double progress) {
  if (node->getToolItem())
    node->getToolItem()->setProgress(progress);
  processedNodes.insert(node);

  // TODO: Implement the ETA feature. A timer updates passed time and remaining time.
  //       This function updates estimates total time and time and date when the operation
  //       will have finished.

  if (boost::dynamic_pointer_cast<Workflow>(node))    // no progress for workflows
    return;

  if (node != progressNode.lock()) {           // new progress
    etaRegression.clear();
    startTime = time(0);
  }

  int passedSeconds = time(0) - startTime;
  etaRegression.addXY(progress, passedSeconds);
  if (etaRegression.haveData()) {
    int totalSeconds = etaRegression.estimateY(100.0);
    int remainingSeconds = totalSeconds - passedSeconds;

    struct tm* timeinfo;
    char buffer[256];
    time_t finishTime = startTime + totalSeconds;

    timeinfo = localtime(&finishTime);
    strftime(buffer, 256, "%b %d %Y %H:%M:%S", timeinfo);

    host::DataModel& model = host::DataModel::getInstance();
    if (model.getPassedLabel())
      model.getPassedLabel()->setText(formatTime(passedSeconds).c_str());
    if (model.getRemainingLabel())
      model.getRemainingLabel()->setText(formatTime(remainingSeconds).c_str());
    if (model.getTotalLabel())
      model.getTotalLabel()->setText(formatTime(totalSeconds).c_str());
    if (model.getFinishedLabel())
      model.getFinishedLabel()->setText(buffer);
  }
  progressNode = node;
}

void WorkbenchWindow::workflowUpdateFinished() {
  for (std::set<boost::weak_ptr<Node> >::iterator iter = processedNodes.begin(); iter != processedNodes.end(); ++iter) {
    if (iter->lock()->getToolItem())
      iter->lock()->getToolItem()->setProgress(ToolItem::Neutral);
  }
  processedNodes.clear();

  Q_EMIT updateFinished();
}

} /* namespace host */

} /* namespace gapputils */
