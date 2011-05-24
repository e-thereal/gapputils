/*
 * Workflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "Workflow.h"

#include <capputils/EnumerableAttribute.h>
#include "Workbench.h"

#include <capputils/ReflectableClassFactory.h>
#include <qtreeview.h>
#include <qsplitter.h>

#include "PropertyGridDelegate.h"
#include "CustomToolItemAttribute.h"
#include "InputsItem.h"
#include "OutputsItem.h"
#include "CableItem.h"
#include <gapputils/HideAttribute.h>
#include <capputils/Xmlizer.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/LibraryLoader.h>
#include <gapputils/WorkflowElement.h>
#include <capputils/Verifier.h>

using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

namespace workflow {

int Workflow::librariesId;

BeginPropertyDefinitions(Workflow)

ReflectableBase(Node)
DefineProperty(Libraries, Enumerable<vector<std::string>*, false>(), Volatile(), Observe(librariesId = PROPERTY_ID))
DefineProperty(Edges, Enumerable<vector<Edge*>*, true>(), Volatile())
DefineProperty(Nodes, Enumerable<vector<Node*>*, true>(), Volatile())
DefineProperty(InputsPosition, Volatile())
DefineProperty(OutputsPosition, Volatile())

EndPropertyDefinitions

Workflow::Workflow() : _InputsPosition(0), _OutputsPosition(0), ownWidget(true), worker(0) {
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
  if (ownWidget)
    delete widget;

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

void Workflow::newModule(const std::string& name) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(name);
  Node* node = new Node();
  node->setModule(object);
  getNodes()->push_back(node);

  newItem(node);
}

TiXmlElement* Workflow::getXml(bool addEmptyModule) const {
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
}

void Workflow::newItem(Node* node) {
  ToolItem* item;
  ICustomToolItemAttribute* customToolItem = node->getModule()->getAttribute<ICustomToolItemAttribute>();

  if (node == &inputsNode)
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
  inputsNode.setModule(getModule());
  outputsNode.setModule(getModule());
  inputsNode.setX(getInputsPosition()[0]);
  inputsNode.setY(getInputsPosition()[1]);
  outputsNode.setX(getOutputsPosition()[0]);
  outputsNode.setY(getOutputsPosition()[1]);

  newItem(&inputsNode);
  newItem(&outputsNode);

  vector<Node*>* nodes = getNodes();
  vector<Edge*>* edges = getEdges();
  for (unsigned i = 0; i < nodes->size(); ++i)
    newItem(nodes->at(i));
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
  buildStack(&outputsNode);
  processStack();
}

void Workflow::processStack() {
  //cout << "[" << QThread::currentThreadId() << "] " << "Process stack" << endl;
  while (!nodeStack.empty()) {
    Node* node = nodeStack.top();
    nodeStack.pop();

    processedStack.push(node);
    // Update the not, if it needs update or if it is the last one
    if (nodeStack.empty() || !node->isUpToDate()) {
      node->getToolItem()->setProgress(-2);
      Q_EMIT processModule(node);
      return;
    } else {
      node->getToolItem()->setProgress(100);
    }
  }
  Q_EMIT updateFinished();

  // set all back
  for(; !processedStack.empty(); processedStack.pop())
    processedStack.top()->getToolItem()->setProgress(-1);
}

void Workflow::finalizeModuleUpdate(Node* node) {
  // write results
  node->writeResults();
  node->getToolItem()->setProgress(100);

  // processStack (will emit updateFinished signal)
  processStack();
}

void Workflow::showProgress(Node* node, int i) {
  node->getToolItem()->setProgress(i);
}

bool Workflow::update(IProgressMonitor* monitor) {
  // Create a new interface module instance
  // Connect outputs to everything the original interface is connected to
  // Update the new interface module as it would be the current node
  // remove all created connections
}

void Workflow::writeResults() {
  // Write all output properties of interace copy to the interface module
}

}

}
