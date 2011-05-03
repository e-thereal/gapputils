#include "WorkflowController.h"

#include <ReflectableClassFactory.h>

#include "CustomToolItemAttribute.h"
#include "Node.h"
#include "DataModel.h"
#include "CableItem.h"

using namespace capputils::reflection;
using namespace std;

namespace gapputils {

using namespace attributes;
using namespace host;

namespace workflow {

Controller* Controller::instance = 0;

Controller::Controller(void) : QObject(), workbench(0)
{
}


Controller::~Controller(void)
{
  if (workbench) {
    disconnect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
    disconnect(workbench, SIGNAL(itemDeleteRequest(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
    disconnect(workbench, SIGNAL(cableCreated(CableItem*)), this, SLOT(createEdge(CableItem*)));
    disconnect(workbench, SIGNAL(cableDeleted(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  }
}

Controller& Controller::getInstance() {
  if (!instance)
    instance = new Controller();

  return *instance;
}

void Controller::setWorkbench(Workbench* workbench) {
  if (this->workbench) {
    disconnect(this->workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
    disconnect(this->workbench, SIGNAL(itemDeleteRequest(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
    disconnect(this->workbench, SIGNAL(cableCreated(CableItem*)), this, SLOT(createEdge(CableItem*)));
    disconnect(this->workbench, SIGNAL(cableDeleted(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  }
  this->workbench = workbench;
  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(itemDeleted(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
  connect(workbench, SIGNAL(cableCreated(CableItem*)), this, SLOT(createEdge(CableItem*)));
  connect(workbench, SIGNAL(cableDeleted(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
}

void Controller::newModule(const std::string& name) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(name);
  Node* node = new Node();
  node->setModule(object);
  DataModel::getInstance().getGraph()->getNodes()->push_back(node);

  newItem(node);
}

void Controller::newItem(Node* node) {
  ToolItem* item;
  ICustomToolItemAttribute* customToolItem = node->getModule()->getAttribute<ICustomToolItemAttribute>();
  if (customToolItem)
    item = customToolItem->createToolItem(node);
  else
    item = new ToolItem(node);

  item->setPos(node->getX(), node->getY());
  node->setToolItem(item);

  workbench->addToolItem(item);
  workbench->setSelectedItem(item);
}

void Controller::newCable(Edge* edge) {
  if (!workbench)
    return;

  DataModel& model = DataModel::getInstance();
  const string outputNodeUuid = edge->getOutputNode();
  const string inputNodeUuid = edge->getInputNode();

  vector<Node*>* nodes = model.getGraph()->getNodes();
  
  Node *outputNode = 0, *inputNode = 0;
  for (unsigned i = 0; i < nodes->size(); ++i)
    if (nodes->at(i)->getUuid().compare(outputNodeUuid) == 0)
      outputNode = nodes->at(i);
    else if (nodes->at(i)->getUuid().compare(inputNodeUuid) == 0)
      inputNode = nodes->at(i);

  ToolConnection *outputConnection = 0, *inputConnection = 0;
  if (outputNode && inputNode) {
    outputConnection = outputNode->getToolItem()->getConnection(edge->getOutputProperty(), ToolConnection::Output);
    inputConnection = inputNode->getToolItem()->getConnection(edge->getOutputProperty(), ToolConnection::Input);

    if (outputConnection && inputConnection) {
      CableItem* cable = new CableItem(workbench, outputConnection, inputConnection);
      workbench->addCableItem(cable);
      edge->setCableItem(cable);
    }
  } else {
    // TODO: Error handling
  }
}

void Controller::resumeFromModel() {
  vector<Node*>* nodes = DataModel::getInstance().getGraph()->getNodes();
  vector<Edge*>* edges = DataModel::getInstance().getGraph()->getEdges();
  for (unsigned i = 0; i < nodes->size(); ++i)
    newItem(nodes->at(i));
  for (unsigned i = 0; i < edges->size(); ++i)
    newCable(edges->at(i));
}

void Controller::itemChangedHandler(ToolItem* item) {
  Node* node = item->getNode();
  node->setX(item->x());
  node->setY(item->y());
}

void Controller::deleteItem(ToolItem* item) {
  vector<Node*>* nodes = DataModel::getInstance().getGraph()->getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i)
    if (nodes->at(i) == item->getNode()) {
      // remove and delete node
      delete nodes->at(i);
      nodes->erase(nodes->begin() + i);
    }
}

void Controller::createEdge(CableItem* cable) {
  Edge* edge = new Edge();
  edge->setOutputNode(cable->getInput()->parent->getNode()->getUuid());
  edge->setOutputProperty(cable->getInput()->property->getName());
  edge->setInputNode(cable->getOutput()->parent->getNode()->getUuid());
  edge->setInputProperty(cable->getOutput()->property->getName());
  edge->setCableItem(cable);

  DataModel::getInstance().getGraph()->getEdges()->push_back(edge);
}

void Controller::deleteEdge(CableItem* cable) {
  vector<Edge*>* edges = DataModel::getInstance().getGraph()->getEdges();
  for (unsigned i = 0; i < edges->size(); ++i)
    if (edges->at(i)->getCableItem() == cable) {
      // remove and delete node
      delete edges->at(i);
      edges->erase(edges->begin() + i);
    }
}

}

}
