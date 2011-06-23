/*
 * Workflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "Workflow.h"

#include <qtreeview.h>
#include <qsplitter.h>

#include <cassert>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/Xmlizer.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/LibraryLoader.h>
#include <capputils/Verifier.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ShortNameAttribute.h>

#include <gapputils/CombinerInterface.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>
#include <gapputils/LabelAttribute.h>

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
DefineProperty(ViewportScale)
DefineProperty(ViewportPosition)

EndPropertyDefinitions

Workflow::Workflow() : _InputsPosition(0), _OutputsPosition(0), _ViewportScale(1.0), ownWidget(true), hasIONodes(false), processingCombination(false), worker(0) {
  _Libraries = new vector<std::string>();
  _Edges = new vector<Edge*>();
  _Nodes = new vector<Node*>();

  _InputsPosition.push_back(500);
  _InputsPosition.push_back(200);

  _OutputsPosition.push_back(1400);
  _OutputsPosition.push_back(200);

  _ViewportPosition.push_back(0);
  _ViewportPosition.push_back(0);

  inputsNode.setUuid("Inputs");
  outputsNode.setUuid("Outputs");

  workbench = new Workbench();
  workbench->setGeometry(0, 0, 600, 600);
  workbench->setChecker(this);

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

  connect(workbench, SIGNAL(createItemRequest(int, int, QString)), this, SLOT(createModule(int, int, QString)));
  connect(workbench, SIGNAL(currentItemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(preItemDeleted(ToolItem*)), this, SLOT(deleteModule(ToolItem*)));
  connect(workbench, SIGNAL(connectionCompleted(CableItem*)), this, SLOT(createEdge(CableItem*)));
  connect(workbench, SIGNAL(connectionRemoved(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  connect(workbench, SIGNAL(viewportChanged()), this, SLOT(handleViewportChanged()));

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

  // Don't delete module before setting it to zero
  // The module property is observed and reflectable. Thus, when resetting
  // the module, and event listener is disconnected from the old module.
  // This will cause the application to crash when the module has already been
  // deleted.
  ReflectableClass* module = getModule();
  setModule(0);
  delete module;

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

void Workflow::createModule(int x, int y, QString classname) {
  if (classname.count() == 0)
    return;

  std::string name = classname.toAscii().data();

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
  node->setX(x);
  node->setY(y);
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

const std::string& getPropertyLabel(IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

void Workflow::newItem(Node* node) {
  ToolItem* item;
  ICustomToolItemAttribute* customToolItem = node->getModule()->getAttribute<ICustomToolItemAttribute>();

  // Get the label corrent label
  string label = string("[") + node->getModule()->getClassName() + "]";
  vector<IClassProperty*>& properties = node->getModule()->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<LabelAttribute>()) {
      label = properties[i]->getStringValue(*node->getModule());
      break;
    }
  }

  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow) {
    item = new WorkflowItem(label);
    connect((WorkflowItem*)item, SIGNAL(showWorkflowRequest(ToolItem*)), this, SLOT(showWorkflow(ToolItem*)));

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
    }
  } else if (node == &inputsNode) {
    item = new ToolItem("Inputs");
    item->setDeletable(false);

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
    }
  } else if (node == &outputsNode) {
    item = new ToolItem("Outputs");
    item->setDeletable(false);

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if (prop->getAttribute<OutputAttribute>())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
    }
  } else if (customToolItem) {
    item = customToolItem->createToolItem(label);
  } else {
    item = new ToolItem(label);

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
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

void Workflow::newCable(Edge* edge) {

//  cout << "Connecting " << edge->getOutputNode() << "." << edge->getOutputProperty()
//       << " with " << edge->getInputNode() << "." << edge->getInputProperty() << "... " << flush;
//
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

    unsigned outputPropertyId, inputPropertyId;
    if (outputNode->getModule()->getPropertyIndex(outputPropertyId, edge->getOutputProperty()) &&
        inputNode->getModule()->getPropertyIndex(inputPropertyId, edge->getInputProperty()))
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
    // TODO: Error handling
    cout << "FAILED!" << endl;
    return;
  }
//  cout << "DONE!" << endl;
}

void Workflow::resumeViewport() {
  vector<double> pos = getViewportPosition();
  workbench->scaleView(getViewportScale());
  workbench->centerOn(pos[0], pos[1]);
  handleViewportChanged();
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
  Node* node = getNode(item);
  if (node)
    propertyGrid->setModel(node->getModel());
}

void Workflow::itemChangedHandler(ToolItem* item) {
  Node* node = getNode(item);
  if (node) {
    node->setX(item->x());
    node->setY(item->y());
  }

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

void Workflow::deleteModule(ToolItem* item) {
  cout << "Deleting module: " << item->getLabel() << endl;

  unsigned i = 0;
  Node* node = getNode(item, i);

  if (!node) {
    cout << "Node not found!" << endl;
    return;
  }

  // delete edges connected to the node
  vector<Edge*>* edges = getEdges();
  for (int j = (int)edges->size() - 1; j >= 0; --j) {
    Edge* edge = edges->at(j);
    if (edge->getInputNodePtr() == node || edge->getOutputNodePtr() == node) {
      removeEdge(edge);
    }
  }

  // remove and delete node
  delete node;
  _Nodes->erase(_Nodes->begin() + i);
}

void Workflow::removeEdge(Edge* edge) {
  vector<Edge*>* edges = getEdges();
  for (unsigned i = 0; i < edges->size(); ++i) {
    if (edges->at(i) == edge) {
      delete edges->at(i);
      edges->erase(edges->begin() + i);
    }
  }
}

void Workflow::createEdge(CableItem* cable) {
  Node* outputNode = getNode(cable->getInput()->parent);
  Node* inputNode = getNode(cable->getOutput()->parent);

  // Sanity check. Should never fail
  if (!outputNode || !outputNode->getModule() || !inputNode || !inputNode->getModule())
    return;

  Edge* edge = new Edge();

  edge->setOutputNode(outputNode->getUuid());
  edge->setOutputProperty(outputNode->getModule()->getProperties()[cable->getInput()->id]->getName());
  edge->setInputNode(inputNode->getUuid());
  edge->setInputProperty(inputNode->getModule()->getProperties()[cable->getOutput()->id]->getName());
  edge->setCableItem(cable);
  if (!edge->activate(outputNode, inputNode)) {
    delete edge;
    workbench->removeCableItem(cable);
  } else {
    getEdges()->push_back(edge);
  }
}

bool Workflow::areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const {
  assert(output);
  assert(input);

  const Node* outputNode = getNode(output->parent);
  const Node* inputNode = getNode(input->parent);

  return Edge::areCompatible(outputNode, output->id, inputNode, input->id);
}

void Workflow::deleteEdge(CableItem* cable) {
  unsigned pos;
  Edge* edge = getEdge(cable, pos);
  if (edge) {
    delete edge;
    _Edges->erase(_Edges->begin() + pos);
  }
}

void Workflow::buildStack(Node* node) {
  // Rebuild the stack without node, thus guaranteeing that node appears only once
  stack<Node*> oldStack;
  while (!nodeStack.empty()) {
    oldStack.push(nodeStack.top());
    nodeStack.pop();
  }
  while (!oldStack.empty()) {
    Node* n = oldStack.top();
    if (n != node)
      nodeStack.push(n);
    oldStack.pop();
  }

  nodeStack.push(node);
  node->getToolItem()->setProgress(-3);

  // call build stack for all output nodes
  vector<Edge*>* edges = getEdges();
  for (int j = (int)edges->size() - 1; j >= 0; --j) {
    Edge* edge = edges->at(j);
    if (edge->getInputNodePtr() == node) {
      buildStack(edge->getOutputNodePtr());
    }
  }
}

void Workflow::updateCurrentModule() {
  //cout << "[" << QThread::currentThreadId() << "] " << "Update selected module" << endl;
  // build stack
  Node* node = getNode(workbench->getCurrentItem());
  if (node) {
    buildStack(node);
    processStack();
  }
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

void Workflow::showWorkflow(ToolItem* item) {
  Node* node = getNode(item);
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow)
    Q_EMIT showWorkflowRequest(workflow);
}

bool Workflow::isUpToDate() const {
  if (dynamic_cast<CombinerInterface*>(getModule()))
    return true;
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
    CableItem* cable = _Edges->at(0)->getCableItem();
    removeEdge(_Edges->at(0));
    if (cable)
      workbench->removeCableItem(cable);
  }

  while (_Nodes->size()) {
    ToolItem* item = _Nodes->at(0)->getToolItem();
    deleteModule(item);
    if (item)
      workbench->removeToolItem(item);
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

Node* Workflow::getNode(ToolItem* item) {
  unsigned pos;

  if (item == inputsNode.getToolItem())
    return &inputsNode;
  else if (item == outputsNode.getToolItem())
    return &outputsNode;
  else
    return getNode(item, pos);
}

Node* Workflow::getNode(ToolItem* item, unsigned& pos) {
  Node* node = 0;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getToolItem() == item) {
      return node;
    }
  }
  return 0;
}

Edge* Workflow::getEdge(CableItem* cable) {
  unsigned pos;
  return getEdge(cable, pos);
}

Edge* Workflow::getEdge(CableItem* cable, unsigned& pos) {
  Edge* edge = 0;
  for(pos = 0; pos < _Edges->size(); ++pos) {
    edge = _Edges->at(pos);
    if (edge->getCableItem() == cable)
      return edge;
  }
  return 0;
}

const Node* Workflow::getNode(ToolItem* item) const {
  unsigned pos;

  if (item == inputsNode.getToolItem())
    return &inputsNode;
  else if (item == outputsNode.getToolItem())
    return &outputsNode;
  else
    return getNode(item, pos);
}

const Node* Workflow::getNode(ToolItem* item, unsigned& pos) const {
  Node* node = 0;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getToolItem() == item) {
      return node;
    }
  }
  return 0;
}

const Edge* Workflow::getEdge(CableItem* cable) const {
  unsigned pos;
  return getEdge(cable, pos);
}

void Workflow::handleViewportChanged() {
  QPointF cnt = workbench->mapToScene(workbench->viewport()->rect().center());

  setViewportScale(workbench->getViewScale());
  vector<double> position;
  position.push_back(cnt.x());
  position.push_back(cnt.y());
  setViewportPosition(position);
}

const Edge* Workflow::getEdge(CableItem* cable, unsigned& pos) const {
  Edge* edge = 0;
  for(pos = 0; pos < _Edges->size(); ++pos) {
    edge = _Edges->at(pos);
    if (edge->getCableItem() == cable)
      return edge;
  }
  return 0;
}

}

}
