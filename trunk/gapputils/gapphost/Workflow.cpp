/*
 * Workflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#define BOOST_FILESYSTEM_VERSION 2

#include "Workflow.h"

#include <qlabel.h>
#include <qclipboard.h>
#include <qapplication.h>

#include <cassert>

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/Xmlizer.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/LibraryLoader.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>
#include <capputils/Logbook.h>

#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/InterfaceAttribute.h>
#include <gapputils/CollectionElement.h>

#include <boost/units/detail/utility.hpp>

#include <set>
#include <map>
#include <iomanip>

#include "CableItem.h"
#include "Workbench.h"
#include "WorkflowItem.h"
#include "PropertyReference.h"

#include "DataModel.h"
#include "MainWindow.h"
#include "LogbookModel.h"

// TODO: shouldn't reference the controller
#include "WorkflowController.h"
#include "WorkflowUpdater.h"

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
DefineProperty(Libraries, Enumerable<Type, false>(), Observe(librariesId = Id))

// Add Properties of node after librarie
// Original reason: module could be an object of a class of one of the libraries)
// Shouldn't be the case anymore but just to be save, leave it here for now. (there's no harm)
ReflectableBase(Node)

DefineProperty(Edges, Enumerable<Type, true>())
DefineProperty(Nodes, Enumerable<Type, true>())
DefineProperty(GlobalProperties, Enumerable<Type, true>())
DefineProperty(GlobalEdges, Enumerable<Type, true>())
DefineProperty(ViewportScale)
DefineProperty(ViewportPosition)
DefineProperty(Logbook, Volatile())

EndPropertyDefinitions

// TODO: If a node changes, check if global properties of the node are still valid.

Workflow::Workflow()
 : _Libraries(new vector<std::string>()),
   _Edges(new vector<boost::shared_ptr<Edge> >()),
   _Nodes(new vector<boost::shared_ptr<Node> >()),
   _GlobalProperties(new vector<boost::shared_ptr<GlobalProperty> >()),
   _GlobalEdges(new vector<boost::shared_ptr<GlobalEdge> >()),
   _ViewportScale(1.0), _Logbook(new Logbook(&host::LogbookModel::GetInstance())),
   ownWidget(true),
   workflowUpdater(new host::WorkflowUpdater())
{
  _ViewportPosition.push_back(0);
  _ViewportPosition.push_back(0);

  _Logbook->setModule("gapputils::workflow::Workflow");

  workbench = new Workbench();
  workbench->setGeometry(0, 0, 600, 600);
  workbench->setChecker(this);
  widget = workbench;

  connect(workbench, SIGNAL(createItemRequest(int, int, QString)), this, SLOT(createModule(int, int, QString)));
  connect(workbench, SIGNAL(currentItemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
  connect(workbench, SIGNAL(itemChanged(ToolItem*)), this, SLOT(itemChangedHandler(ToolItem*)));
  connect(workbench, SIGNAL(preItemDeleted(ToolItem*)), this, SLOT(deleteModule(ToolItem*)));
  connect(workbench, SIGNAL(connectionCompleted(CableItem*)), this, SLOT(createEdge(CableItem*)));
  connect(workbench, SIGNAL(connectionRemoved(CableItem*)), this, SLOT(deleteEdge(CableItem*)));
  connect(workbench, SIGNAL(viewportChanged()), this, SLOT(handleViewportChanged()));

  connect(workflowUpdater.get(), SIGNAL(progressed(boost::shared_ptr<workflow::Node>, double)), this, SLOT(showProgress(boost::shared_ptr<workflow::Node>, double)));
  connect(workflowUpdater.get(), SIGNAL(updateFinished()), this, SLOT(workflowUpdateFinished()));

  this->Changed.connect(EventHandler<Workflow>(this, &Workflow::changedHandler));
}

#define TRACE std::cout << __LINE__ << std::endl;

Workflow::~Workflow() {
//  std::cout << "Deleting workflow." << std::endl;

  Q_EMIT deleteCalled(getUuid());

  LibraryLoader& loader = LibraryLoader::getInstance();

//  if (ownWidget) {
//    delete widget;
//  }

  // Clean up before releasing the libraries
  _Edges->clear();
  _GlobalEdges->clear();
  _Nodes->clear();
  _GlobalProperties->clear();
  setModule(boost::shared_ptr<ReflectableClass>());

  // Unload libraries
  for (unsigned i = 0; i < _Libraries->size(); ++i) {
    loader.freeLibrary(_Libraries->at(i));
  }

  map<string, boost::weak_ptr<Workflow> >& workflowMap = *host::DataModel::getInstance().getWorkflowMap();
  if (workflowMap.find(getUuid()) != workflowMap.end())
    workflowMap.erase(getUuid());
}

void Workflow::addInterfaceNode(boost::shared_ptr<Node> node) {
  Logbook& dlog = *getLogbook();
  interfaceNodes.push_back(node);

  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);

  IClassProperty* prop = object->findProperty("Value");
  if (!prop)
    return;

  if (prop->getAttribute<InputAttribute>()) {
    ToolItem* item = getToolItem();
    if (!item) {
      dlog(Severity::Trace) << "Workflow does not have a ToolItem";
    } else {
      item->addConnection(QString(object->getProperty("Label").c_str()), interfaceNodes.size() + getModule()->getProperties().size() - 1, ToolConnection::Output);
//      std::cout << "[Info] New connection added with id " << interfaceNodes.size() + getModule()->getProperties().size() - 1 << std::endl;
    }
  }
  if (prop->getAttribute<OutputAttribute>()) {
    ToolItem* item = getToolItem();
    if (!item) {
      dlog(Severity::Trace) << "Workflow does not have a ToolItem";
    } else {
      item->addConnection(QString(object->getProperty("Label").c_str()), interfaceNodes.size() + getModule()->getProperties().size() - 1, ToolConnection::Input);
//      std::cout << "[Info] New connection added with id " << interfaceNodes.size() + getModule()->getProperties().size() - 1 << std::endl;
    }
  }
}
  
void Workflow::removeInterfaceNode(boost::shared_ptr<Node> node) {
  Logbook& dlog = *getLogbook();
//  std::cout << "[Info] removing interface node" << std::endl;

  int deletedId = -1;
  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);

  IClassProperty* prop = object->findProperty("Value");

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i] == node) {
      interfaceNodes.erase(interfaceNodes.begin() + i);
//      std::cout << "[Info] removed idx = " << i << std::endl;

      // delete edges connected to the node/tool connection
      boost::shared_ptr<Workflow> parent = getWorkflow().lock();
      if (parent) {
        boost::shared_ptr<vector<boost::shared_ptr<Edge> > > edges = parent->getEdges();
        for (int j = (int)edges->size() - 1; j >= 0; --j) {
          boost::shared_ptr<Edge> edge = edges->at(j);
          if (edge->getInputProperty() == node->getUuid() || edge->getOutputProperty() == node->getUuid()) {
            if (edge->getCableItem())
              parent->workbench->removeCableItem(edge->getCableItem());
            parent->removeEdge(edge);
          }
        }
      }

      // remove connection with ID = module->propertyCount() + i
      if (prop) {
        if (prop->getAttribute<InputAttribute>()) {
          ToolItem* item = getToolItem();
          if (!item) {
            dlog(Severity::Trace) << "[Info] Workflow does not have a ToolItem";
          } else {
            deletedId = getModule()->getProperties().size() + i;
//            std::cout << "[Info] removedId = " << deletedId << std::endl;
            item->deleteConnection(deletedId, ToolConnection::Output);
          }
        }

        if (prop->getAttribute<OutputAttribute>()) {
          ToolItem* item = getToolItem();
          if (!item) {
            dlog(Severity::Trace) << "[Info] Workflow does not have a ToolItem";
          } else {
            deletedId = getModule()->getProperties().size() + i;
//            std::cout << "[Info] removedId = " << deletedId << std::endl;
            item->deleteConnection(deletedId, ToolConnection::Input);
          }
        }
      }
      break;
    }
  }

  // decrement all IDs of all connections whoes IDs are greater then the deleted ID
//  std::cout << "[Info] Decrementing Ids which are greater than " << deletedId << std::endl;
  ToolItem* item = getToolItem();
  if (!item) {
//    std::cout << "[Info] Workflow does not have a ToolItem" << std::endl;
  } else {
    std::vector<boost::shared_ptr<ToolConnection> > outputs;
    item->getOutputs(outputs);

    for (unsigned i = 0; i < outputs.size(); ++i) {
//      std::cout << "[Info] Id = " << outputs[i]->id << std::endl;
      if (outputs[i]->id > deletedId) {
//        std::cout << "[Info] Found toolconnection. Current Id = " << outputs[i]->id << std::endl;
        --(outputs[i]->id);
//        std::cout << "[Info] New id = " << outputs[i]->id << std::endl;
        if (outputs[i]->multi) {
//          std::cout << "[Info] Changed multi id from " << outputs[i]->multi->id;
          outputs[i]->multi->id = outputs[i]->id;
//          std::cout << " to " << outputs[i]->multi->id << std::endl;
        }
      }
    }
    std::vector<boost::shared_ptr<ToolConnection> >& inputs = item->getInputs();

    for (unsigned i = 0; i < inputs.size(); ++i) {
//      std::cout << "[Info] Id = " << inputs[i]->id << std::endl;
      if (inputs[i]->id > deletedId) {
//        std::cout << "[Info] Found toolconnection. Current Id = " << inputs[i]->id << std::endl;
        --(inputs[i]->id);
//        std::cout << "[Info] New id = " << inputs[i]->id << std::endl;
        if (inputs[i]->multi) {
//          std::cout << "[Info] Changed multi id from " << inputs[i]->multi->id;
          inputs[i]->multi->id = inputs[i]->id;
//          std::cout << " to " << inputs[i]->multi->id << std::endl;
        }
      }
    }
  }
}

bool Workflow::hasCollectionElementInterface() const {
  boost::shared_ptr<CollectionElement> collection;
  for (unsigned i = 0; i < interfaceNodes.size(); ++i)
    if ((collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i]->getModule())) && collection->getCalculateCombinations())
      return true;
  return false;
}

boost::shared_ptr<const Node> Workflow::getInterfaceNode(int id) const {
  assert(getModule());
  const int pos = id - getModule()->getProperties().size();
  if (pos >= 0 && (unsigned)pos < interfaceNodes.size())
    return interfaceNodes[pos];
  return boost::shared_ptr<Node>();
}

std::vector<boost::shared_ptr<Node> >& Workflow::getInterfaceNodes() {
  return interfaceNodes;
}

void Workflow::makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference) {
  boost::shared_ptr<GlobalProperty> globalProperty(new GlobalProperty());
  globalProperty->setName(name);
  globalProperty->setModuleUuid(propertyReference.getNodeId());
  globalProperty->setPropertyId(propertyReference.getPropertyId());
  getGlobalProperties()->push_back(globalProperty);
}

void Workflow::connectProperty(const std::string& name, const PropertyReference& propertyReference) {
  // Find global property instance by name
  boost::shared_ptr<GlobalProperty> globalProp = getGlobalProperty(name);

  boost::shared_ptr<GlobalEdge> edge(new GlobalEdge());
  edge->setOutputNode(globalProp->getModuleUuid());
  edge->setOutputProperty(globalProp->getPropertyId());
  edge->setInputNode(propertyReference.getNodeId());
  edge->setInputProperty(propertyReference.getPropertyId());
  edge->setGlobalProperty(name);

  getGlobalEdges()->push_back(edge);
  activateGlobalEdge(edge);
}

void Workflow::activateGlobalEdge(boost::shared_ptr<GlobalEdge> edge) {
  Logbook& dlog = *getLogbook();
  boost::shared_ptr<Node> inputNode = getNode(edge->getInputNode());

  boost::shared_ptr<GlobalProperty> globalProp = getGlobalProperty(edge->getGlobalProperty());
  globalProp->addEdge(edge);
  if (!edge->activate(getNode(edge->getOutputNode()), inputNode)) {
    dlog(Severity::Error) << "Error in line " << __LINE__;
  }
}

void Workflow::removeGlobalProperty(boost::shared_ptr<GlobalProperty> gprop) {
  // remove edges and expressions to that property first
  while(gprop->getEdges()->size()) {
    removeGlobalEdge(boost::dynamic_pointer_cast<GlobalEdge>(gprop->getEdges()->at(0).lock()));
  }

  while(gprop->getExpressions()->size()) {
    gprop->getExpressions()->at(0).lock()->disconnect(gprop);
  }

  for (unsigned i = 0; i < _GlobalProperties->size(); ++i) {
    if (_GlobalProperties->at(i) == gprop) {
      _GlobalProperties->erase(_GlobalProperties->begin() + i);
      break;
    }
  }
}

void Workflow::removeGlobalEdge(boost::shared_ptr<GlobalEdge> edge) {
  boost::shared_ptr<GlobalProperty> gprop = getGlobalProperty(edge->getGlobalProperty());
  assert(gprop);
  gprop->removeEdge(edge);
  for (unsigned i = 0; i < _GlobalEdges->size(); ++i) {
    if (_GlobalEdges->at(i) == edge) {
      _GlobalEdges->erase(_GlobalEdges->begin() + i);
      break;
    }
  }
}

void addDependencies(boost::shared_ptr<Workflow> workflow, const std::string& classname) {
  // Update libraries
  string libName = LibraryLoader::getInstance().classDefinedIn(classname);
  if (libName.size()) {
    boost::shared_ptr<vector<string> > libraries = workflow->getLibraries();
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

//void Workflow::createModule(int x, int y, QString classname) {
//  if (classname.count() == 0)
//    return;
//
//  std::string name = classname.toAscii().data();
//
//  boost::shared_ptr<ReflectableClass> object = boost::shared_ptr<ReflectableClass>(ReflectableClassFactory::getInstance().newInstance(name));
//  addDependencies(shared_from_this(), name);
//
//  boost::shared_ptr<Node> node;
//  if (boost::dynamic_pointer_cast<WorkflowInterface>(object)) {
//    boost::shared_ptr<Workflow> workflow = boost::shared_ptr<Workflow>(new Workflow());
//    workflow->setModule(object);
//    addDependencies(workflow, name);
//    workflow->resume();
//    connect(workflow.get(), SIGNAL(deleteCalled(const std::string&)), this, SLOT(delegateDeleteCalled(const std::string&)));
//    connect(workflow.get(), SIGNAL(showWorkflowRequest(boost::shared_ptr<workflow::Workflow>)), this, SLOT(showWorkflow(boost::shared_ptr<workflow::Workflow>)));
//    node = workflow;
//  } else {
//    node = boost::shared_ptr<Node>(new Node());
//    node->setModule(object);
//    node->resume();
//  }
//  node->setWorkflow(shared_from_this());
//  node->setX(x);
//  node->setY(y);
//  getNodes()->push_back(node);
//
//  if (object->getAttribute<InterfaceAttribute>()) {
//    addInterfaceNode(node);
//  }
//
//  newItem(node);
//}

const std::string& getPropertyLabel(IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

//void Workflow::newItem(boost::shared_ptr<Node> node) {
//  ToolItem* item;
//  assert(node->getModule());
//
//  // Get the label
//  string label = string("[") + node->getModule()->getClassName() + "]";
//  vector<IClassProperty*>& properties = node->getModule()->getProperties();
//  for (unsigned i = 0; i < properties.size(); ++i) {
//    if (properties[i]->getAttribute<LabelAttribute>()) {
//      label = properties[i]->getStringValue(*node->getModule());
//      break;
//    }
//  }
//
//  boost::shared_ptr<Workflow> workflow = boost::dynamic_pointer_cast<Workflow>(node);
//  if (workflow) {
//    item = new WorkflowItem(label);
//    connect((WorkflowItem*)item, SIGNAL(showWorkflowRequest(ToolItem*)), this, SLOT(showWorkflow(ToolItem*)));
//
//    for (unsigned i = 0; i < properties.size(); ++i) {
//      IClassProperty* prop = properties[i];
//      if (prop->getAttribute<InputAttribute>() && !prop->getAttribute<FromEnumerableAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
//      if (prop->getAttribute<OutputAttribute>() && !prop->getAttribute<ToEnumerableAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
//    }
//  } else if (node->getModule()->getAttribute<InterfaceAttribute>()) {
//    item = new ToolItem(label);
//    connect(item, SIGNAL(showDialogRequested(ToolItem*)), this, SLOT(showModuleDialog(ToolItem*)));
//
//    // Search for the value property and add it. Don't add any other stuff
//    for (unsigned i = 0; i < properties.size(); ++i) {
//      IClassProperty* prop = properties[i];
//      //if (prop->getName() != "Value")
//      //  continue;
//      if (prop->getAttribute<InputAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
//      if (prop->getAttribute<OutputAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
//      //break;
//    }
//  } else {
//    item = new ToolItem(label);
//    connect(item, SIGNAL(showDialogRequested(ToolItem*)), this, SLOT(showModuleDialog(ToolItem*)));
//
//    for (unsigned i = 0; i < properties.size(); ++i) {
//      IClassProperty* prop = properties[i];
//      if (prop->getAttribute<InputAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
//      if (prop->getAttribute<OutputAttribute>())
//        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
//    }
//  }
//
//  item->setPos(node->getX(), node->getY());
//  node->setToolItem(item);
//
//  workbench->addToolItem(item);
//  workbench->setCurrentItem(item);
//}

bool Workflow::resumeEdge(boost::shared_ptr<Edge> edge) {
  Logbook& dlog = *getLogbook();

//  cout << "Connecting " << edge->getOutputNode() << "." << edge->getOutputProperty()
//       << " with " << edge->getInputNode() << "." << edge->getInputProperty() << "... " << flush;
//
  const string outputNodeUuid = edge->getOutputNode();
  const string inputNodeUuid = edge->getInputNode();

  vector<boost::shared_ptr<Node> >& nodes = *getNodes();

  boost::shared_ptr<Node> outputNode, inputNode;

  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (nodes[i]->getUuid().compare(outputNodeUuid) == 0)
      outputNode = nodes[i];
    if (nodes[i]->getUuid().compare(inputNodeUuid) == 0)
      inputNode = nodes[i];
  }

  if (outputNode && inputNode) {
    edge->activate(outputNode, inputNode);
  } else {
    dlog(Severity::Warning) << "Can not find connections for edge '" << edge->getInputNode() << "' -> '" << edge->getOutputNode() << "'";
    return false;
  }
//  cout << "DONE!" << endl;
  return true;
}

void Workflow::resumeViewport() {
  vector<double> pos = getViewportPosition();
  workbench->setViewScale(getViewportScale());
  workbench->centerOn(pos[0], pos[1]);
  handleViewportChanged();
}

//string replaceAll(const string& context, const string& from, const string& to)
//{
//  string str = context;
//  size_t lookHere = 0;
//  size_t foundHere;
//  while((foundHere = str.find(from, lookHere)) != string::npos)
//  {
//    str.replace(foundHere, from.size(), to);
//        lookHere = foundHere + to.size();
//  }
//  return str;
//}

void Workflow::resume() {
  Logbook& dlog = *getLogbook();

  map<string, boost::weak_ptr<Workflow> >& workflowMap = *host::DataModel::getInstance().getWorkflowMap();
  //assert(workflowMap->find(getUuid()) == workflowMap->end());
  if (workflowMap.find(getUuid()) == workflowMap.end())
    workflowMap.insert(pair<string, boost::shared_ptr<Workflow> >(getUuid(), shared_from_this()));

  vector<boost::shared_ptr<Node> >& nodes = *getNodes();
  vector<boost::shared_ptr<Edge> >& edges = *getEdges();
  vector<boost::shared_ptr<GlobalProperty> >& globals = *getGlobalProperties();
  vector<boost::shared_ptr<GlobalEdge> >& gedges = *getGlobalEdges();

  for (unsigned i = 0; i < nodes.size(); ++i)
    resumeNode(nodes[i]);

  for (unsigned i = 0; i < edges.size(); ++i) {
    if (!resumeEdge(edges[i])) {
      removeEdge(edges[i]);
      --i;
      dlog() << "Edge has been removed from the model.";
    }
  }

  for (unsigned i = 0; i < gedges.size(); ++i)
    activateGlobalEdge(gedges[i]);

  for (unsigned i = 0; i < nodes.size(); ++i)
    nodes[i]->resumeExpressions();

  // reset output checksum if at least one volatile output node
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    IClassProperty* prop = interfaceNodes[i]->getModule()->findProperty("Value");
    if (prop && prop->getAttribute<InputAttribute>() && prop->getAttribute<VolatileAttribute>()) {
      setOutputChecksum(0);
      break;
    }
    prop = interfaceNodes[i]->getModule()->findProperty("Values");
    if (prop && prop->getAttribute<InputAttribute>() && prop->getAttribute<VolatileAttribute>()) {
      setOutputChecksum(0);
      break;
    }
  }
}

void Workflow::resumeNode(boost::shared_ptr<Node> node) {
  node->setWorkflow(shared_from_this());
//  newItem(node);
  node->resume();
  boost::shared_ptr<Workflow> workflow = boost::dynamic_pointer_cast<Workflow>(node);
  if (workflow) {
    connect(workflow.get(), SIGNAL(deleteCalled(const std::string&)), this, SLOT(delegateDeleteCalled(const std::string&)));
    connect(workflow.get(), SIGNAL(showWorkflowRequest(boost::shared_ptr<workflow::Workflow>)), this, SLOT(showWorkflow(boost::shared_ptr<workflow::Workflow>)));
  }

  if (node->getModule() && node->getModule()->getAttribute<InterfaceAttribute>())
    addInterfaceNode(node);
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
    boost::shared_ptr<vector<string> > libraries = getLibraries();
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
  boost::shared_ptr<Node> node = getNode(item);
  Q_EMIT currentModuleChanged(node);
}

void Workflow::itemChangedHandler(ToolItem* item) {
  boost::shared_ptr<Node> node = getNode(item);
  if (node) {
    node->setX(item->x());
    node->setY(item->y());
  }
}

void Workflow::deleteModule(ToolItem* item) {
//  cout << "Deleting module: " << item->getLabel() << endl;
  Logbook& dlog = *getLogbook();

  unsigned i = 0;
  boost::shared_ptr<Node> node = getNode(item, i);

  if (!node) {
    dlog(Severity::Error) << "Node not found! " << __FILE__ << ", " << __LINE__;
    return;
  }

  if (node->getModule() && node->getModule()->getAttribute<InterfaceAttribute>()) {
    removeInterfaceNode(node);
  }

  // delete global edges connected to the node
  vector<boost::shared_ptr<GlobalEdge> >& gedges = *getGlobalEdges();
  for (int j = (int)gedges.size() - 1; j >= 0; --j) {
    boost::shared_ptr<GlobalEdge> edge = gedges[j];
    if (edge->getInputNode() == node->getUuid()) {
      removeGlobalEdge(edge);
    }
  }

  // delete global properties of the node
  vector<boost::shared_ptr<GlobalProperty> >& gprops = *getGlobalProperties();
  for (int j = (int)gprops.size() - 1; j >= 0; --j) {
    boost::shared_ptr<GlobalProperty> gprop = gprops[j];
    if (gprop->getModuleUuid() == node->getUuid()) {
      removeGlobalProperty(gprop);
    }
  }

  // delete edges connected to the node
  vector<boost::shared_ptr<Edge> >& edges = *getEdges();
  for (int j = (int)edges.size() - 1; j >= 0; --j) {
    boost::shared_ptr<Edge> edge = edges[j];
    if (edge->getInputNode() == node->getUuid() || edge->getOutputNode() == node->getUuid()) {
      removeEdge(edge);
    }
  }

  // remove and delete node
//  delete node;
  _Nodes->erase(_Nodes->begin() + i);
}

std::string Workflow::getPropertyName(boost::shared_ptr<const Node> node, int connectionId) const {
  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);
  if (connectionId < 0)
    return "";

  int propertyCount = object->getProperties().size();
  boost::shared_ptr<const Workflow> workflow = boost::dynamic_pointer_cast<const Workflow>(node);

  if (connectionId < propertyCount) {
    return object->getProperties()[connectionId]->getName();
  } else if (workflow && (unsigned)connectionId < propertyCount + workflow->interfaceNodes.size()) {
    return workflow->interfaceNodes[connectionId - propertyCount]->getUuid();
  } else {
    return "";
  }
}

bool Workflow::getToolConnectionId(boost::shared_ptr<const Node> node, const std::string& propertyName, unsigned& id) const {
  assert(node);
  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);

  if (object->getPropertyIndex(id, propertyName))
    return true;

  boost::shared_ptr<const Workflow> workflow = boost::dynamic_pointer_cast<const Workflow>(node);
  if (workflow) {
    const std::vector<boost::shared_ptr<Node> >& interfaceNodes = workflow->interfaceNodes;
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (interfaceNodes[i]->getUuid() == propertyName) {
        id = object->getProperties().size() + i;
        return true;
      }
    }
  }

  return false;
}
//boost::enable_shared_from_this<Workflow>::
void Workflow::removeEdge(boost::shared_ptr<Edge> edge) {
  boost::shared_ptr<vector<boost::shared_ptr<Edge> > > edges = getEdges();
  for (unsigned i = 0; i < edges->size(); ++i) {
    if (edges->at(i) == edge) {
      edges->erase(edges->begin() + i);
    }
  }
}

void Workflow::createEdge(CableItem* cable) {
  boost::shared_ptr<Node> outputNode = getNode(cable->getInput()->parent);
  boost::shared_ptr<Node> inputNode = getNode(cable->getOutput()->parent);

  // Sanity check. Should never fail
  assert(outputNode && outputNode->getModule() && inputNode && inputNode->getModule());

  boost::shared_ptr<Edge> edge(new Edge());

  edge->setOutputNode(outputNode->getUuid());
  edge->setOutputProperty(getPropertyName(outputNode, cable->getInput()->id));
  edge->setInputNode(inputNode->getUuid());
  edge->setInputProperty(getPropertyName(inputNode, cable->getOutput()->id));

  edge->setCableItem(cable);
  if (!edge->activate(outputNode, inputNode)) {
    workbench->removeCableItem(cable);
  } else {
    getEdges()->push_back(edge);
  }
}

bool Workflow::areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const {
  assert(output);
  assert(input);

  // TODO: use property references here

  boost::shared_ptr<const Node> outputNode = getNode(output->parent);
  unsigned outputId = output->id;
  if (output->id >= (int)outputNode->getModule()->getProperties().size()) {
    // Get interface node and ID of value property
    boost::shared_ptr<const Workflow> workflow = boost::dynamic_pointer_cast<const Workflow>(outputNode);
    if (workflow) {
      outputNode = workflow->getInterfaceNode(output->id);
      assert(outputNode->getModule());
      if (boost::dynamic_pointer_cast<CollectionElement>(outputNode->getModule())) {
        if (!outputNode->getModule()->getPropertyIndex(outputId, "Values"))
          outputNode.reset();
      } else {
        if (!outputNode->getModule()->getPropertyIndex(outputId, "Value"))
          outputNode.reset();
      }
    } else {
      outputNode.reset();
    }
  }

  boost::shared_ptr<const Node> inputNode = getNode(input->parent);
  unsigned inputId = input->id;
  if (input->id >= (int)inputNode->getModule()->getProperties().size()) {
    // Get interface node and ID of Value property
    boost::shared_ptr<const Workflow> workflow = boost::dynamic_pointer_cast<const Workflow>(inputNode);
    if (workflow) {
      inputNode = workflow->getInterfaceNode(input->id);
      assert(inputNode->getModule());
      if (boost::dynamic_pointer_cast<CollectionElement>(inputNode->getModule())) {
        if (!inputNode->getModule()->getPropertyIndex(inputId, "Values"))
          inputNode.reset();
      } else {
        if (!inputNode->getModule()->getPropertyIndex(inputId, "Value"))
          inputNode.reset();
      }
    } else {
      inputNode.reset();
    }
  }


  capputils::reflection::IClassProperty* inProp = inputNode->getModule()->getProperties()[inputId];
  capputils::reflection::IClassProperty* outProp = outputNode->getModule()->getProperties()[outputId];

  return Edge::areCompatible(inProp, outProp);
}

void Workflow::deleteEdge(CableItem* cable) {
  unsigned pos;
  boost::shared_ptr<Edge> edge = getEdge(cable, pos);
  if (edge) {
    _Edges->erase(_Edges->begin() + pos);
  }
}

bool Workflow::isInputNode(boost::shared_ptr<const Node> node) const {
  if (!node)
    return false;

  boost::shared_ptr<ReflectableClass> module = node->getModule();

  if (!module)
    return false;

  return module->getAttribute<InterfaceAttribute>() && module->findProperty("Value")->getAttribute<OutputAttribute>();
}

bool Workflow::isOutputNode(boost::shared_ptr<const Node> node) const {
  if (!node)
    return false;

  boost::shared_ptr<ReflectableClass> module = node->getModule();

  if (!module)
    return false;

  return module->getAttribute<InterfaceAttribute>() && module->findProperty("Value")->getAttribute<InputAttribute>();
}

void Workflow::getDependentNodes(boost::shared_ptr<Node> node, std::vector<boost::shared_ptr<Node> >& dependendNodes) {
  // If input node see to which node of the parent workflow this node is connected
  if (isInputNode(node)) {
    boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
    if (workflow) {
      {vector<boost::shared_ptr<Edge> >& edges = *workflow->getEdges();
      for (unsigned i = 0; i < edges.size(); ++i) {
        boost::shared_ptr<Edge> edge = edges[i];
        if (edge->getInputNode() == getUuid() && edge->getInputProperty() == node->getUuid())
          dependendNodes.push_back(workflow->getNode(edge->getOutputNode()));
      }}

      {vector<boost::shared_ptr<GlobalEdge> >& gedges = *workflow->getGlobalEdges();
      for (unsigned i = 0; i < gedges.size(); ++i) {
        boost::shared_ptr<GlobalEdge> gedge = gedges[i];
        if (gedge->getInputNode() == getUuid() && gedge->getInputProperty() == node->getUuid())
          dependendNodes.push_back(workflow->getNode(gedge->getOutputNode()));
      }}
    }
  } else {
    vector<boost::shared_ptr<Edge> >& edges = *getEdges();
    for (unsigned i = 0; i < edges.size(); ++i) {
      boost::shared_ptr<Edge> edge = edges[i];
      if (edge->getInputNode() == node->getUuid()) {
        // TODO: check that the property does not have the NoParameter attribute
        dependendNodes.push_back(getNode(edge->getOutputNode()));
      }
    }

    vector<boost::shared_ptr<GlobalEdge> >& gedges = *getGlobalEdges();
    for (unsigned i = 0; i < gedges.size(); ++i) {
      boost::shared_ptr<GlobalEdge> gedge = gedges[i];
      if (gedge->getInputNode() == node->getUuid()) {
        // TODO: check that the property does not have the NoParameter attribute
        dependendNodes.push_back(getNode(gedge->getOutputNode()));
      }
    }
  }
}

bool Workflow::isDependentProperty(boost::shared_ptr<const Node> node, const std::string& propertyName) const {
  if (isInputNode(node)) {
    boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
    if (workflow && propertyName == "Value") {
      boost::shared_ptr<vector<boost::shared_ptr<Edge> > > edges = workflow->getEdges();
      for (unsigned i = 0; i < edges->size(); ++i) {
        boost::shared_ptr<Edge> edge = edges->at(i);
        if (edge->getInputNode() == getUuid() && edge->getInputProperty() == node->getUuid())
          return true;
      }

      boost::shared_ptr<vector<boost::shared_ptr<GlobalEdge> > > gedges = workflow->getGlobalEdges();
      for (unsigned i = 0; i < gedges->size(); ++i) {
        boost::shared_ptr<GlobalEdge> gedge = gedges->at(i);
        if (gedge->getInputNode() == getUuid() && gedge->getInputProperty() == node->getUuid())
          return true;
      }
    }
  } else {
    boost::shared_ptr<vector<boost::shared_ptr<Edge> > > edges = getEdges();
    for (unsigned i = 0; i < edges->size(); ++i) {
      boost::shared_ptr<Edge> edge = edges->at(i);
      if (edge->getInputNode() == node->getUuid() && edge->getInputProperty() == propertyName)
        return true;
    }

    boost::shared_ptr<vector<boost::shared_ptr<GlobalEdge> > > gedges = getGlobalEdges();
    for (unsigned i = 0; i < gedges->size(); ++i) {
      boost::shared_ptr<GlobalEdge> gedge = gedges->at(i);
      if (gedge->getInputNode() == node->getUuid() && gedge->getInputProperty() == propertyName)
        return true;
    }
  }
  PropertyReference ref(shared_from_this(), node->getUuid(), propertyName);
  boost::shared_ptr<const CollectionElement> collection = boost::dynamic_pointer_cast<const CollectionElement>(node->getModule());
  if (ref.getProperty() && ref.getProperty()->getAttribute<FromEnumerableAttribute>() && collection && collection->getCalculateCombinations())
    return true;

  return false;
}

void Workflow::updateCurrentModule() {
  
  // build stack
  boost::shared_ptr<Node> node = getNode(workbench->getCurrentItem());
  if (!node)
    return;

  // update checksums before updating the workflow
  workflowUpdater->update(node);
}

void Workflow::workflowUpdateFinished() {
  for (std::set<boost::weak_ptr<Node> >::iterator iter = processedNodes.begin(); iter != processedNodes.end(); ++iter)
    iter->lock()->getToolItem()->setProgress(ToolItem::Neutral);
  processedNodes.clear();

  Q_EMIT updateFinished(shared_from_this());
}

boost::shared_ptr<Node> Workflow::getCurrentNode() {
  return getNode(workbench->getCurrentItem());
}

void Workflow::updateOutputs() {
  workflowUpdater->update(shared_from_this());
}

void Workflow::abortUpdate() {
  workflowUpdater->abort();
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

  out << setfill('0');
  if (days) {
    out << days << "d ";
    out << setw(2) << hours << "h ";
    out << setw(2) << minutes << "min ";
    out << setw(2) << seconds << "s";
  } else if (hours) {
    out << hours << "h ";
    out << setw(2) << minutes << "min ";
    out << setw(2) << seconds << "s";
  } else if (minutes) {
    out << minutes << "min ";
    out << setw(2) << seconds << "s";
  } else {
    out << seconds << "s";
  }

  // maximum length is 19.
  return out.str() + std::string(25 - out.str().size(), ' ');
}

void Workflow::showProgress(boost::shared_ptr<Node> node, double progress) {
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

void Workflow::showWorkflow(boost::shared_ptr<Workflow> workflow) {
  Q_EMIT showWorkflowRequest(workflow);
}

void Workflow::showWorkflow(ToolItem* item) {
  boost::shared_ptr<Node> node = getNode(item);
  boost::shared_ptr<Workflow> workflow = boost::dynamic_pointer_cast<Workflow>(node);
  if (workflow)
    Q_EMIT showWorkflowRequest(workflow);
}

void Workflow::showModuleDialog(ToolItem* item) {
  boost::shared_ptr<Node> node = getNode(item);
  boost::shared_ptr<WorkflowElement> element = boost::dynamic_pointer_cast<WorkflowElement>(node->getModule());
  if (element)
    element->show();
}

void Workflow::delegateDeleteCalled(const std::string& uuid) {
//  std::cout << "delegate delete called" << std::endl;
  Q_EMIT deleteCalled(uuid);
}

void Workflow::copySelectedNodesToClipboard() {
  Workflow copyWorkflow;
  std::set<std::string> copied;

  // Temporarily add nodes to the node list for the xmlization.
  // Nodes have to be removed afterwards in order to avoid a double free memory
  boost::shared_ptr<std::vector<boost::shared_ptr<Node> > > nodes = copyWorkflow.getNodes();
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems()) {
    ToolItem* toolItem = dynamic_cast<ToolItem*>(item);
    if (toolItem) {
      boost::shared_ptr<Node> node = getNode(toolItem);
      // TODO: Workflows are not copied unless renewUuid() is fully implemented
      if (!boost::dynamic_pointer_cast<Workflow>(node)) {
        nodes->push_back(node);
        copied.insert(node->getUuid());
      }
    }
  }

  // Add all edges to the workflow where both ends nodes are about to be copied
  boost::shared_ptr<std::vector<boost::shared_ptr<Edge> > > edges = copyWorkflow.getEdges();
  for (unsigned i = 0; i < getEdges()->size(); ++i) {
    boost::shared_ptr<Edge> edge = getEdges()->at(i);
    if (copied.find(edge->getInputNode()) != copied.end() &&
        copied.find(edge->getOutputNode()) != copied.end())
    {
      edges->push_back(edge);
    }
  }

  std::stringstream xmlStream;
  Xmlizer::ToXml(xmlStream, copyWorkflow);

  nodes->clear();
  edges->clear();

  QApplication::clipboard()->setText(xmlStream.str().c_str());
}

void renewUuids(Workflow& workflow) {
  std::map<std::string, std::string> uuidMap;

  // Go through edges, nodes, global properties, global edges
  std::vector<boost::shared_ptr<Node> >& nodes = *workflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    Node& node = *nodes[i];
    const std::string uuid = node.getUuid();
    if (uuidMap.find(uuid) == uuidMap.end())
      uuidMap[uuid] = Node::CreateUuid();
    node.setUuid(uuidMap[uuid]);

    // TODO: Not implemented unless the ID change is applied to all
    // occurances of an UUID in the workflow
//    Workflow* subworkflow = dynamic_cast<Workflow*>(&node);
//    if (subworkflow)
//      renewUuids(*subworkflow);
  }

  std::vector<boost::shared_ptr<Edge> >& edges = *workflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    Edge& edge = *edges[i];

    // Change IDs only if mapping is available. If no mapping is availabe,
    // the ID will likely not need a change (input or output node IDs)
    if (uuidMap.find(edge.getInputNode()) != uuidMap.end())
      edge.setInputNode(uuidMap[edge.getInputNode()]);
    if (uuidMap.find(edge.getOutputNode()) != uuidMap.end())
      edge.setOutputNode(uuidMap[edge.getOutputNode()]);
  }

  // TODO: replace UUIDs of other parts as well
}

//void Workflow::addNodesFromClipboard() {
//  Logbook& dlog = *getLogbook();
//
//  Workflow pasteWorkflow;
//  const std::string clipboardText = QApplication::clipboard()->text().toUtf8().data();
//  Xmlizer::FromXmlString(pasteWorkflow, clipboardText);
//
//  // Unselect selected items
//  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems())
//    item->setSelected(false);
//
//  renewUuids(pasteWorkflow);
//  std::vector<boost::shared_ptr<Node> >& nodes = *pasteWorkflow.getNodes();
//  for (unsigned i = 0; i < nodes.size(); ++i) {
//    getNodes()->push_back(nodes[i]);
//    resumeNode(nodes[i]);
//    nodes[i]->getToolItem()->setSelected(true);
//  }
//  nodes.clear(); // avoid double free memory
//
//  std::vector<boost::shared_ptr<Edge> >& edges = *pasteWorkflow.getEdges();
//  for (unsigned i = 0; i < edges.size(); ++i) {
//    getEdges()->push_back(edges[i]);
//    if (!newCable(edges[i])) {
//      removeEdge(edges[i]);
//      dlog() << "Edge has been removed from the model." << endl;
//    }
//  }
//
//  edges.clear(); // avoid double free memory
//}

void Workflow::setUiEnabled(bool enabled) {
  workbench->setModifiable(enabled);

  boost::shared_ptr<vector<boost::shared_ptr<Node> > > nodes = getNodes();
  for (unsigned i = 0; i < nodes->size(); ++i) {
    boost::shared_ptr<Workflow> workflow = boost::dynamic_pointer_cast<Workflow>(nodes->at(i));
    if (workflow)
      workflow->setUiEnabled(enabled);
  }
}

boost::shared_ptr<Node> Workflow::getNode(ToolItem* item) {
  unsigned pos;
  return getNode(item, pos);
}

boost::shared_ptr<Node> Workflow::getNode(ToolItem* item, unsigned& pos) {
  boost::shared_ptr<Node> node;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getToolItem() == item) {
      return node;
    }
  }
  return boost::shared_ptr<Node>();
}

boost::shared_ptr<Node> Workflow::getNode(boost::shared_ptr<capputils::reflection::ReflectableClass> object) {
  unsigned pos;
  return getNode(object, pos);
}

boost::shared_ptr<Node> Workflow::getNode(boost::shared_ptr<capputils::reflection::ReflectableClass> object, unsigned& pos) {
  boost::shared_ptr<Node> node;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getModule() == object) {
      return node;
    }
  }
  return boost::shared_ptr<Node>();
}

boost::shared_ptr<Node> Workflow::getNode(const std::string& uuid) const {
  boost::shared_ptr<Node> node;

  for(unsigned pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getUuid().compare(uuid) == 0) {
      return node;
    }
  }
  return boost::shared_ptr<Node>();
}

boost::shared_ptr<Edge> Workflow::getEdge(CableItem* cable) {
  unsigned pos;
  return getEdge(cable, pos);
}

boost::shared_ptr<Edge> Workflow::getEdge(CableItem* cable, unsigned& pos) {
  boost::shared_ptr<Edge> edge;
  for(pos = 0; pos < _Edges->size(); ++pos) {
    edge = _Edges->at(pos);
    if (edge->getCableItem() == cable)
      return edge;
  }
  return boost::shared_ptr<Edge>();
}

boost::shared_ptr<const Node> Workflow::getNode(ToolItem* item) const {
  unsigned pos;
  return getNode(item, pos);
}

boost::shared_ptr<const Node> Workflow::getNode(ToolItem* item, unsigned& pos) const {
  boost::shared_ptr<Node> node;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getToolItem() == item) {
      return node;
    }
  }
  return boost::shared_ptr<Node>();
}

boost::shared_ptr<const Edge> Workflow::getEdge(CableItem* cable) const {
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

boost::shared_ptr<const Edge> Workflow::getEdge(CableItem* cable, unsigned& pos) const {
  boost::shared_ptr<Edge> edge;
  for(pos = 0; pos < _Edges->size(); ++pos) {
    edge = _Edges->at(pos);
    if (edge->getCableItem() == cable)
      return edge;
  }
  return boost::shared_ptr<Edge>();
}

boost::shared_ptr<GlobalProperty> Workflow::getGlobalProperty(const std::string& name) {
  boost::shared_ptr<GlobalProperty> property;

  for(unsigned pos = 0; pos < _GlobalProperties->size(); ++pos) {
    property = _GlobalProperties->at(pos);
    if (property->getName().compare(name) == 0)
      return property;
  }
  return boost::shared_ptr<GlobalProperty>();
}

boost::shared_ptr<GlobalProperty> Workflow::getGlobalProperty(const PropertyReference& reference)
{
  boost::shared_ptr<GlobalProperty> gprop;

  for(unsigned pos = 0; pos < _GlobalProperties->size(); ++pos) {
    gprop = _GlobalProperties->at(pos);
    if (gprop->getModuleUuid() == reference.getNodeId() && gprop->getPropertyId() == reference.getPropertyId())
      return gprop;
  }
  return boost::shared_ptr<GlobalProperty>();
}

// TODO: re-think how to identify a global edge
boost::shared_ptr<GlobalEdge> Workflow::getGlobalEdge(const PropertyReference& reference)
{
  boost::shared_ptr<GlobalEdge> edge;
  for(unsigned pos = 0; pos < _GlobalEdges->size(); ++pos) {
    edge = _GlobalEdges->at(pos);
    if (edge->getInputNode() == reference.getNodeId() &&
        edge->getInputProperty() == reference.getPropertyId())
    {
      return edge;
    }
  }
  return boost::shared_ptr<GlobalEdge>();
}

bool Workflow::trySelectNode(const std::string& uuid) {
  boost::shared_ptr<Node> node = getNode(uuid);
  if (node && node->getToolItem()) {
    workbench->setExclusivelySelected(node->getToolItem());
    workbench->setCurrentItem(node->getToolItem());
    return true;
  }

  return false;
}

void Workflow::resetInputs() {
  std::vector<boost::shared_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i]->getModule());
    if (collection && isInputNode(interfaceNodes[i]))
      collection->resetCombinations();
  }
}

void Workflow::incrementInputs() {
  std::vector<boost::shared_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i]->getModule());
    if (collection && isInputNode(interfaceNodes[i]))
      collection->advanceCombinations();
  }
}

void Workflow::decrementInputs() {
  std::vector<boost::shared_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i]->getModule());
    if (collection && isInputNode(interfaceNodes[i]))
      collection->regressCombinations();
  }
}

}

}
