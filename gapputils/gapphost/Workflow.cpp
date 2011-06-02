/*
 * Workflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "Workflow.h"

#include <qtreeview.h>
#include <qsplitter.h>

#include <capputils/ReflectableClassFactory.h>
#include <capputils/Xmlizer.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/LibraryLoader.h>
#include <capputils/Verifier.h>
#include <capputils/EnumerableAttribute.h>

#include <gapputils/CombinerInterface.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>

#include "PropertyGridDelegate.h"
#include "CustomToolItemAttribute.h"
#include "InputsItem.h"
#include "OutputsItem.h"
#include "CableItem.h"
#include "Workbench.h"
#include "WorkflowItem.h"

using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

namespace workflow {

int Workflow::librariesId;

BeginPropertyDefinitions(Workflow)

// Libraries must be the first property since libraries must be loaded before all other modules
DefineProperty(Libraries, Enumerable<vector<std::string>*, false>(), Observe(librariesId = PROPERTY_ID))

// Add Properties of node after libraries (module could be an object of a class of one of the libraries)
ReflectableBase(Node)

DefineProperty(Edges, Enumerable<vector<Edge*>*, true>())
DefineProperty(Nodes, Enumerable<vector<Node*>*, true>())
DefineProperty(InputsPosition)
DefineProperty(OutputsPosition)

EndPropertyDefinitions

Workflow::Workflow() : _InputsPosition(0), _OutputsPosition(0), ownWidget(true), hasIONodes(false), processingCombination(false), worker(0) {
  _Libraries = new vector<std::string>();
  _Edges = new vector<Edge*>();
  _Nodes = new vector<Node*>();

  _InputsPosition.push_back(-230);
  _InputsPosition.push_back(-230);

  _OutputsPosition.push_back(170);
  _OutputsPosition.push_back(-230);

  inputsNode.setUuid("Inputs");
  outputsNode.setUuid("Outputs");

  workbench = new Workbench();
  workbench->setGeometry(0, 0, 600, 600);

  propertyGrid = new QTreeView();
  propertyGrid->setAllColumnsShowFocus(false);
  propertyGrid->setAlternatingRowColors(true);
  propertyGrid->setSelectionBehavior(QAbstractItemView::SelectItems);
  propertyGrid->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  propertyGrid->setItemDelegate(new PropertyGridDelegate());

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(workbench);
  splitter->addWidget(propertyGrid);
  splitter->setSizes(QList<int>() << 900 << 260);
  widget = splitter;

  connect(workbench, SIGNAL(itemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(itemDeleted(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
  connect(workbench, SIGNAL(cableCreated(CableItem*)), this, SLOT(createEdge(CableItem*)));
  connect(workbench, SIGNAL(cableDeleted(CableItem*)), this, SLOT(deleteEdge(CableItem*)));

  worker = new WorkflowWorker(this);
  worker->start();

  this->Changed.connect(EventHandler<Workflow>(this, &Workflow::changedHandler));
}

Workflow::~Workflow() {
  Q_EMIT deleteCalled(this);

  LibraryLoader& loader = LibraryLoader::getInstance();

  if (worker) {
    worker->quit();
    worker->wait();
    delete worker;
  }

  disconnect(workbench, SIGNAL(itemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
  disconnect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  disconnect(workbench, SIGNAL(itemDeleted(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
  disconnect(workbench, SIGNAL(cableCreated(CableItem*)), this, SLOT(createEdge(CableItem*)));
  disconnect(workbench, SIGNAL(cableDeleted(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  //workbench->scene()->clear();
  if (ownWidget) {
    delete widget;
    inputsNode.setToolItem(0);
    outputsNode.setToolItem(0);
  }

  for (unsigned i = 0; i < _Edges->size(); ++i)
    delete _Edges->at(i);
  delete _Edges;

  for (unsigned i = 0; i < _Nodes->size(); ++i)
    delete _Nodes->at(i);
  delete _Nodes;

  inputsNode.setModule(0);
  outputsNode.setModule(0);

  ReflectableClass* module = getModule();
  if (module) {
    setModule(0);
    delete module;
  }

  // Unload libraries
  for (unsigned i = 0; i < _Libraries->size(); ++i)
    loader.freeLibrary(_Libraries->at(i));
  delete _Libraries;
}

void addDependencies(Workflow* workflow, const std::string& classname) {
  // Update libraries
  string libName = LibraryLoader::getInstance().classDefinedIn(classname);
  if (libName.size()) {
    vector<string>* libraries = workflow->getLibraries();
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

void Workflow::newModule(const std::string& name) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(name);
  addDependencies(this, name);

  Node* node = 0;
  if (dynamic_cast<WorkflowInterface*>(object)) {
    Workflow* workflow = new Workflow();
    workflow->setModule(object);
    addDependencies(workflow, name);
    workflow->resumeFromModel();
    connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(delegateDeleteCalled(workflow::Workflow*)));
    connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
    node = workflow;
  } else {
    node = new Node();
    node->setModule(object);
  }
  node->setX(workbench->scene()->sceneRect().width() / 2);
  node->setY(workbench->scene()->sceneRect().height() / 2);
  getNodes()->push_back(node);

  newItem(node);
}

/*TiXmlElement* Workflow::getXml(bool addEmptyModule) const {
  TiXmlElement* element = new TiXmlElement(makeXmlName(getClassName()));
  Xmlizer::AddPropertyToXml(*element, *this, findProperty("Libraries"));
  Xmlizer::AddPropertyToXml(*element, *this, findProperty("Edges"));
  Xmlizer::AddPropertyToXml(*element, *this, findProperty("Nodes"));
  Xmlizer::AddPropertyToXml(*element, *this, findProperty("InputsPosition"));
  Xmlizer::AddPropertyToXml(*element, *this, findProperty("OutputsPosition"));
  if (addEmptyModule) {
    TiXmlElement* moduleElement = new TiXmlElement("Module");
    moduleElement->LinkEndChild(new TiXmlElement(makeXmlName(getModule()->getClassName())));
    element->LinkEndChild(moduleElement);
  }
  return element;
}*/

void Workflow::newItem(Node* node) {
  ToolItem* item;
  ICustomToolItemAttribute* customToolItem = node->getModule()->getAttribute<ICustomToolItemAttribute>();
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow) {
    item = new WorkflowItem(node);
    connect((WorkflowItem*)item, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
  } else if (node == &inputsNode)
    item = new InputsItem(node);
  else if (node == &outputsNode)
    item = new OutputsItem(node);
  else if (customToolItem)
    item = customToolItem->createToolItem(node);
  else
    item = new ToolItem(node);

  item->setPos(node->getX(), node->getY());
  node->setToolItem(item);

  workbench->addToolItem(item);
  workbench->setSelectedItem(item);
}

void Workflow::newCable(Edge* edge) {

  const string outputNodeUuid = edge->getOutputNode();
  const string inputNodeUuid = edge->getInputNode();

  vector<Node*>* nodes = getNodes();

  Node *outputNode = 0, *inputNode = 0;

  if (inputsNode.getUuid().compare(outputNodeUuid) == 0)
    outputNode = &inputsNode;
  if (outputsNode.getUuid().compare(inputNodeUuid) == 0)
    inputNode = &outputsNode;

  for (unsigned i = 0; i < nodes->size(); ++i) {
    if (nodes->at(i)->getUuid().compare(outputNodeUuid) == 0)
      outputNode = nodes->at(i);
    if (nodes->at(i)->getUuid().compare(inputNodeUuid) == 0)
      inputNode = nodes->at(i);
  }

  ToolConnection *outputConnection = 0, *inputConnection = 0;
  if (outputNode && inputNode) {
    outputConnection = outputNode->getToolItem()->getConnection(edge->getOutputProperty(), ToolConnection::Output);
    inputConnection = inputNode->getToolItem()->getConnection(edge->getInputProperty(), ToolConnection::Input);

    if (outputConnection && inputConnection) {
      CableItem* cable = new CableItem(workbench, outputConnection, inputConnection);
      workbench->addCableItem(cable);
      edge->setCableItem(cable);
    }
  } else {
    // TODO: Error handling
  }
}

void Workflow::resumeFromModel() {
  if (!hasIONodes) {
    hasIONodes = true;
    inputsNode.setModule(getModule());
    outputsNode.setModule(getModule());
    inputsNode.setX(getInputsPosition()[0]);
    inputsNode.setY(getInputsPosition()[1]);
    outputsNode.setX(getOutputsPosition()[0]);
    outputsNode.setY(getOutputsPosition()[1]);

    newItem(&inputsNode);
    newItem(&outputsNode);
  }

  vector<Node*>* nodes = getNodes();
  vector<Edge*>* edges = getEdges();
  for (unsigned i = 0; i < nodes->size(); ++i) {
    Workflow* workflow = dynamic_cast<Workflow*>(nodes->at(i));
    if (workflow) {
      workflow->resumeFromModel();
      connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(delegateDeleteCalled(workflow::Workflow*)));
      connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
    }
    newItem(nodes->at(i));
  }
  for (unsigned i = 0; i < edges->size(); ++i)
    newCable(edges->at(i));
}

QWidget* Workflow::dispenseWidget() {
  ownWidget = false;
  return widget;
}

void Workflow::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == librariesId) {
    //cout << "Libraries updated." << endl;
    LibraryLoader& loader = LibraryLoader::getInstance();
    set<string> unusedLibraries = loadedLibraries;
    vector<string>* libraries = getLibraries();
    for (unsigned i = 0; i < libraries->size(); ++i) {

      const string& lib = libraries->at(i);

      // Check if new library
      if (loadedLibraries.find(lib) == loadedLibraries.end()) {
        loader.loadLibrary(lib);
        loadedLibraries.insert(lib);
      }
      // Check if library is still marked for unloading
      set<string>::iterator pos = unusedLibraries.find(lib);
      if (pos != unusedLibraries.end())
        unusedLibraries.erase(pos);
    }

    // Unload all unused libraries
    for (set<string>::iterator pos = unusedLibraries.begin();
        pos != unusedLibraries.end(); ++pos)
    {
      loader.freeLibrary(*pos);
    }
  }
}

void Workflow::itemSelected(ToolItem* item) {
  propertyGrid->setModel(item->getModel());
}

void Workflow::itemChangedHandler(ToolItem* item) {
  Node* node = item->getNode();
  node->setX(item->x());
  node->setY(item->y());

  if (node == &inputsNode) {
    vector<int> pos;
    pos.push_back(item->x());
    pos.push_back(item->y());
    setInputsPosition(pos);
  } else if (node == &outputsNode) {
    vector<int> pos;
    pos.push_back(item->x());
    pos.push_back(item->y());
    setOutputsPosition(pos);
  }
}

void Workflow::deleteItem(ToolItem* item) {
  vector<Node*>* nodes = getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i)
    if (nodes->at(i) == item->getNode()) {
      // remove and delete node
      delete nodes->at(i);
      nodes->erase(nodes->begin() + i);
      break;
    }
}

void Workflow::createEdge(CableItem* cable) {
  Edge* edge = new Edge();
  edge->setOutputNode(cable->getInput()->parent->getNode()->getUuid());
  edge->setOutputProperty(cable->getInput()->property->getName());
  edge->setInputNode(cable->getOutput()->parent->getNode()->getUuid());
  edge->setInputProperty(cable->getOutput()->property->getName());
  edge->setCableItem(cable);

  getEdges()->push_back(edge);
}

void Workflow::deleteEdge(CableItem* cable) {
  vector<Edge*>* edges = getEdges();
  for (unsigned i = 0; i < edges->size(); ++i)
    if (edges->at(i)->getCableItem() == cable) {
      // remove and delete node
      delete edges->at(i);
      edges->erase(edges->begin() + i);
    }
}

void Workflow::buildStack(Node* node) {
  nodeStack.push(node);
  node->getToolItem()->setProgress(-3);

  // call build stack for all input connected nodes
  vector<ToolConnection*>& inputs = node->getToolItem()->getInputs();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    CableItem* cable = inputs[i]->cable;
    if (cable && cable->getInput()) {
      Node* newNode = cable->getInput()->parent->getNode();
      buildStack(newNode);
    }
  }
}

void Workflow::updateSelectedModule() {
  //cout << "[" << QThread::currentThreadId() << "] " << "Update selected module" << endl;
  // build stack and set all to red
  //workbench->getSelectedItem()->getNode()->setUpToDate(false);
  buildStack(workbench->getSelectedItem()->getNode());
  processStack();
}

void Workflow::updateOutputs() {
  // if multiple interface
  //  - clear multiple outputs
  //  - reset combinations iterator
  CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(getModule());
  if (combiner) {
    combiner->resetCombinations();
    processingCombination = true;
    ToolItem* item = getToolItem();
    if (item)
      item->setProgress(5);
  }

  buildStack(&outputsNode);
  processStack();
}

void Workflow::processStack() {
  //cout << "[" << QThread::currentThreadId() << "] " << "Process stack" << endl;
  while (!nodeStack.empty()) {
    Node* node = nodeStack.top();
    nodeStack.pop();

    processedStack.push(node);

    // Update the node, if it needs update or if it is the last one
    if (nodeStack.empty() || !node->isUpToDate()) {
      node->getToolItem()->setProgress(-2);
      Workflow* workflow = dynamic_cast<Workflow*>(node);
      if (workflow) {
        connect(workflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(finalizeModuleUpdate(workflow::Node*)));
        workflow->updateOutputs();
      } else {
        Q_EMIT processModule(node);
      }
      return;
    } else {
      node->getToolItem()->setProgress(100);
    }
  }

  // set all back
  for(; !processedStack.empty(); processedStack.pop())
    processedStack.top()->getToolItem()->setProgress(-1);

  // if multiple interface:
  //  - Add single results to multiple outputs
  //  - if not done
  //     * advance to next calculation
  //     * Rebuild stack
  //     * Start calculation (process stack)
  CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(getModule());
  if (combiner && processingCombination) {
    combiner->appendResults();
    if (combiner->advanceCombinations()) {
      ToolItem* item = getToolItem();
      if (item)
        item->setProgress(combiner->getProgress());
      buildStack(&outputsNode);
      processStack();
      return;       // return here. otherwise update finished is emitted.
    } else {
      processingCombination = false;
    }
  }

  Q_EMIT updateFinished(this);
}

void Workflow::finalizeModuleUpdate(Node* node) {
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow) {
    disconnect(workflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(finalizeModuleUpdate(workflow::Node*)));
  } else {
    node->writeResults();
  }
  node->getToolItem()->setProgress(100);

  // processStack (will emit updateFinished signal)
  processStack();
}

void Workflow::showProgress(Node* node, int i) {
  node->getToolItem()->setProgress(i);
}

void Workflow::showWorkflow(Workflow* workflow) {
  Q_EMIT showWorkflowRequest(workflow);
}

bool Workflow::isUpToDate() const {
  return false;
}

void Workflow::update(IProgressMonitor* monitor) {
}

void Workflow::writeResults() {
}

void Workflow::delegateDeleteCalled(workflow::Workflow* workflow) {
  Q_EMIT deleteCalled(workflow);
}

void Workflow::load(const string& filename) {
  // Delete all current nodes and edges
  // Load model data from xml file (only selected properties)
  // resume
  while (_Edges->size()) {
    workbench->removeCableItem(_Edges->at(0)->getCableItem());
  }

  while (_Nodes->size()) {
    workbench->removeToolItem(_Nodes->at(0)->getToolItem());
  }

  Xmlizer::GetPropertyFromXml(*this, findProperty("Libraries"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("Nodes"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("Edges"), filename);
  resumeFromModel();
}

void Workflow::setUiEnabled(bool enabled) {
  propertyGrid->setEnabled(enabled);
  workbench->setModifiable(enabled);

  vector<Node*>* nodes = getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i) {
    Workflow* workflow = dynamic_cast<Workflow*>(nodes->at(i));
    if (workflow)
      workflow->setUiEnabled(enabled);
  }
}

}

}
