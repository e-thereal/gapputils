#include "WorkflowController.h"

#include <ReflectableClassFactory.h>

#include "CustomToolItemAttribute.h"
#include "Node.h"
#include "DataModel.h"

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
  }
  this->workbench = workbench;
  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(itemDeleteRequest(ToolItem*)), this, SLOT(deleteItem(ToolItem*)));
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

  item->setX(node->getX());
  item->setY(node->getY());

  workbench->addToolItem(item);
  workbench->setSelectedItem(item);
}

void Controller::resumeFromModel() {
  vector<Node*>* nodes = DataModel::getInstance().getGraph()->getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i)
    newItem(nodes->at(i));
}

void Controller::itemChangedHandler(ToolItem* item) {
  Node* node = item->getNode();
  node->setX(item->x());
  node->setY(item->y());
}

void Controller::deleteItem(ToolItem* item) {
  if (!workbench)
    return;

  workbench->scene()->removeItem(item);
  vector<Node*>* nodes = DataModel::getInstance().getGraph()->getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i)
    if (nodes->at(i) == item->getNode()) {
      // remove and delete node
      delete nodes->at(i);
      nodes->erase(nodes->begin() + i);
    }
  delete item;
}

}

}
