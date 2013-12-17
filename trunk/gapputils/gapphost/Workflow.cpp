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
#include <qfile.h>

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
#include <algorithm>

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
int Workflow::globalPropertiesId;
int Workflow::globalEdgesId;

BeginPropertyDefinitions(Workflow)

// Libraries must be the first property since libraries must be loaded before all other modules
DefineProperty(Libraries, Enumerable<Type, false>(), Observe(librariesId = Id))

// Add Properties of node after librarie
// Original reason: module could be an object of a class of one of the libraries)
// Shouldn't be the case anymore but just to be save, leave it here for now. (there's no harm)
ReflectableBase(Node)

DefineProperty(Edges, Enumerable<Type, true>())
DefineProperty(Nodes, Enumerable<Type, true>())
DefineProperty(GlobalProperties, Enumerable<Type, true>(), Observe(globalPropertiesId = Id))
DefineProperty(GlobalEdges, Enumerable<Type, true>(), Observe(globalEdgesId = Id))
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
   _ViewportScale(1.0), _Logbook(new Logbook(&host::LogbookModel::GetInstance()))//,
{
  _ViewportPosition.push_back(0);
  _ViewportPosition.push_back(0);

  _Logbook->setModule("gapputils::workflow::Workflow");
  this->Changed.connect(EventHandler<Workflow>(this, &Workflow::changedHandler));
}

#define TRACE std::cout << __LINE__ << std::endl;

Workflow::~Workflow() {
//  std::cout << "Deleting workflow." << std::endl;
  LibraryLoader& loader = LibraryLoader::getInstance();

  // Clean up before releasing the libraries
  _Edges->clear();
  _GlobalEdges->clear();
  _Nodes->clear();
  _GlobalProperties->clear();
  setModule(boost::shared_ptr<ReflectableClass>());

  // Unload libraries
  for (unsigned i = 0; i < _Libraries->size(); ++i) {
    loader.unloadLibrary(_Libraries->at(i));
  }

  host::DataModel& model = host::DataModel::getInstance();
  boost::shared_ptr<map<string, boost::weak_ptr<Workflow> > > workflowMap = model.getWorkflowMap();
  if (workflowMap->find(getUuid()) != workflowMap->end()) {
    workflowMap->erase(getUuid());
    model.setWorkflowMap(workflowMap);
  }
}

void Workflow::addInterfaceNode(boost::shared_ptr<Node> node) {
  // Only add if not already added
  for (size_t i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i].lock()->getUuid() == node->getUuid())
      return;
  }

  interfaceNodes.push_back(node);

  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);

  node->Changed.connect(EventHandler<Workflow>(this, &Workflow::interfaceChangedHandler));

  IClassProperty* prop = object->findProperty("Value");
  if (!prop)
    return;

  if (prop->getAttribute<InputAttribute>()) {
    ToolItem* item = getToolItem();
    if (item)
      item->addConnection(QString(object->getProperty("Label").c_str()), node->getUuid(), ToolConnection::Output, false);
  }
  if (prop->getAttribute<OutputAttribute>()) {
    ToolItem* item = getToolItem();
    if (item)
      item->addConnection(QString(object->getProperty("Label").c_str()), node->getUuid(), ToolConnection::Input, true);
  }
}

void Workflow::removeInterfaceNode(boost::shared_ptr<Node> node) {
//  std::cout << "[Info] removing interface node" << std::endl;

//  int deletedId = -1;
  boost::shared_ptr<ReflectableClass> object = node->getModule();
  assert(object);

  IClassProperty* prop = object->findProperty("Value");

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i].lock() == node) {
      interfaceNodes.erase(interfaceNodes.begin() + i);

      // delete edges connected to the node/tool connection
      boost::shared_ptr<Workflow> parent = getWorkflow().lock();
      if (parent) {
        boost::shared_ptr<vector<boost::shared_ptr<Edge> > > edges = parent->getEdges();
        for (int j = (int)edges->size() - 1; j >= 0; --j) {
          boost::shared_ptr<Edge> edge = edges->at(j);
          if (edge->getInputProperty() == node->getUuid() || edge->getOutputProperty() == node->getUuid()) {
            parent->removeEdge(edge);
          }
        }
      }

      if (prop) {
        if (prop->getAttribute<InputAttribute>()) {
          ToolItem* item = getToolItem();
          if (!item) {
            //dlog(Severity::Trace) << "[Info] Workflow does not have a ToolItem";
          } else {
            item->deleteConnection(node->getUuid(), ToolConnection::Output);
          }
        }

        if (prop->getAttribute<OutputAttribute>()) {
          ToolItem* item = getToolItem();
          if (!item) {
            //dlog(Severity::Trace) << "[Info] Workflow does not have a ToolItem";
          } else {
            item->deleteConnection(node->getUuid(), ToolConnection::Input);
          }
        }
      }
      break;
    }
  }
}

void Workflow::moveInterfaceNode(int from, int to) {
  assert(from >= 0);
  assert(to >= 0);
  assert(from < (int)interfaceNodes.size());
  assert(to < (int)interfaceNodes.size());

  boost::weak_ptr<Node> temp;
  boost::shared_ptr<Node> fromNode = interfaceNodes[from].lock();
  boost::shared_ptr<Node> toNode = interfaceNodes[to].lock();
  boost::shared_ptr<Node> temp2;
  std::vector<boost::shared_ptr<Node> >& nodes = *getNodes();

  int nodesFrom = -1, nodesTo = -1;

  for (size_t iNode = 0; iNode < nodes.size(); ++iNode) {
    if (nodes[iNode] == fromNode)
      nodesFrom = iNode;
    if (nodes[iNode] == toNode)
      nodesTo = iNode;
  }
  assert(nodesFrom >= 0);
  assert(nodesTo >= 0);

  if (from < to) {
    for (int iNode = from; iNode + 1 <= to; ++iNode) {
      temp = interfaceNodes[iNode];
      interfaceNodes[iNode] = interfaceNodes[iNode + 1];
      interfaceNodes[iNode + 1] = temp;
    }

    for (int iNode = nodesFrom; iNode + 1 <= nodesTo; ++iNode) {
      temp2 = nodes[iNode];
      nodes[iNode] = nodes[iNode + 1];
      nodes[iNode + 1] = temp2;
    }


  } else {
    for (int iNode = from; iNode - 1 >= to; --iNode) {
      temp = interfaceNodes[iNode];
      interfaceNodes[iNode] = interfaceNodes[iNode - 1];
      interfaceNodes[iNode - 1] = temp;
    }

    for (int iNode = nodesFrom; iNode - 1 >= nodesTo; --iNode) {
      temp2 = nodes[iNode];
      nodes[iNode] = nodes[iNode - 1];
      nodes[iNode - 1] = temp2;
    }

  }
}

bool Workflow::hasCollectionElementInterface() const {
  boost::shared_ptr<CollectionElement> collection;
  for (unsigned i = 0; i < interfaceNodes.size(); ++i)
    if ((collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i].lock()->getModule())) && collection->getCalculateCombinations())
      return true;
  return false;
}

//boost::shared_ptr<const Node> Workflow::getInterfaceNode(int id) const {
//  assert(getModule());
//  const int pos = id - getModule()->getProperties().size();
//  if (pos >= 0 && (unsigned)pos < interfaceNodes.size())
//    return interfaceNodes[pos].lock();
//  return boost::shared_ptr<Node>();
//}

std::vector<boost::weak_ptr<Node> >& Workflow::getInterfaceNodes() {
  return interfaceNodes;
}

void Workflow::makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference) {
  boost::shared_ptr<GlobalProperty> globalProperty(new GlobalProperty());
  globalProperty->setName(name);
  globalProperty->setModuleUuid(propertyReference.getNodeId());
  globalProperty->setPropertyId(propertyReference.getPropertyId());
  getGlobalProperties()->push_back(globalProperty);
  setGlobalProperties(getGlobalProperties());     // trigger changed event
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
  setGlobalEdges(getGlobalEdges());               // trigger changed event
}

bool Workflow::activateGlobalEdge(boost::shared_ptr<GlobalEdge> edge) {
  boost::shared_ptr<Node> inputNode = getNode(edge->getInputNode());

  boost::shared_ptr<GlobalProperty> globalProp = getGlobalProperty(edge->getGlobalProperty());
  if (!globalProp)
    return false;

  if (!edge->activate(getNode(edge->getOutputNode()), inputNode)) {
    return false;
  }
  globalProp->addEdge(edge);
  return true;
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
  setGlobalProperties(getGlobalProperties());
}

void Workflow::removeGlobalEdge(boost::shared_ptr<GlobalEdge> edge) {
  boost::shared_ptr<GlobalProperty> gprop = getGlobalProperty(edge->getGlobalProperty());
  if(gprop)
    gprop->removeEdge(edge);
  for (unsigned i = 0; i < _GlobalEdges->size(); ++i) {
    if (_GlobalEdges->at(i) == edge) {
      _GlobalEdges->erase(_GlobalEdges->begin() + i);
      break;
    }
  }
  setGlobalEdges(getGlobalEdges());
}

boost::shared_ptr<Edge> Workflow::createEdge(const PropertyReference& fromProperty, const PropertyReference& toProperty) {
  boost::shared_ptr<Edge> edge(new Edge());
  edge->setOutputNode(fromProperty.getNodeId());
  edge->setOutputProperty(fromProperty.getPropertyId());
  edge->setInputNode(toProperty.getNodeId());
  edge->setInputProperty(toProperty.getPropertyId());
  _Edges->push_back(edge);
  assert(resumeEdge(edge));

  return edge;
}

const std::string& getPropertyLabel(IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

bool Workflow::resumeEdge(boost::shared_ptr<Edge> edge) {
  Logbook& dlog = *getLogbook();

  // Update input positions
  // Only edges that are in the edge list can be resumed
  std::vector<boost::shared_ptr<Edge> >& edges = *getEdges();
  assert(std::find(edges.begin(), edges.end(), edge) != edges.end());
  for (size_t iEdge = 0, pos = 0; iEdge < edges.size(); ++iEdge) {
    if (edges[iEdge]->getInputNode() == edge->getInputNode() &&
        edges[iEdge]->getInputProperty() == edge->getInputProperty())
    {
      edges[iEdge]->setInputPosition(pos++);
    }
  }

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

  if (outputNode && inputNode && edge->activate(outputNode, inputNode))
    return true;

  dlog(Severity::Warning) << "Can not find connections for edge '" << edge->getInputNode() << "' -> '" << edge->getOutputNode() << "'";
  return false;
}

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
      dlog(Severity::Warning) << "Edge has been removed from the model.";
    }
  }

  for (unsigned i = 0; i < globals.size(); ++i) {
    boost::shared_ptr<GlobalProperty> gprop = globals[i];
    if (!PropertyReference::TryCreate(
        shared_from_this(), gprop->getModuleUuid(), gprop->getPropertyId()))
    {
      globals.erase(globals.begin() + i);
      --i;
      dlog.setUuid(gprop->getModuleUuid());
      dlog(Severity::Warning) << "Global property '" << gprop->getName() << "' has been removed from the model: "
          << gprop->getPropertyId();
      dlog.setUuid("");
    }
  }

  for (unsigned i = 0; i < gedges.size(); ++i) {
    if (!activateGlobalEdge(gedges[i])) {
      removeGlobalEdge(gedges[i]);
      --i;
      dlog() << "Global edge has been removed from the model.";
    }
  }

  for (unsigned i = 0; i < nodes.size(); ++i)
    nodes[i]->resumeExpressions();

  // reset output checksum if at least one volatile output node
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    IClassProperty* prop = interfaceNodes[i].lock()->getModule()->findProperty("Value");
    if (prop && prop->getAttribute<InputAttribute>() && prop->getAttribute<VolatileAttribute>()) {
      setOutputChecksum(0);
      break;
    }
    prop = interfaceNodes[i].lock()->getModule()->findProperty("Values");
    if (prop && prop->getAttribute<InputAttribute>() && prop->getAttribute<VolatileAttribute>()) {
      setOutputChecksum(0);
      break;
    }
  }
}

void Workflow::identifyInterfaceNodes() {
  vector<boost::shared_ptr<Node> >& nodes = *getNodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i]->getModule() && nodes[i]->getModule()->getAttribute<InterfaceAttribute>())
      addInterfaceNode(nodes[i]);
  }
}

void Workflow::resumeNode(boost::shared_ptr<Node> node) {
  node->setWorkflow(shared_from_this());
  node->resume();

  if (node->getModule() && node->getModule()->getAttribute<InterfaceAttribute>())
    addInterfaceNode(node);
}

void Workflow::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == librariesId) {
    //cout << "Libraries updated." << endl;
    LibraryLoader& loader = LibraryLoader::getInstance();
    set<string> unusedLibraries = loadedLibraries;
    boost::shared_ptr<vector<string> > libraries = getLibraries();
    const std::string path = host::DataModel::getInstance().getLibraryPath();
    for (unsigned i = 0; i < libraries->size(); ++i) {

      const string& lib = libraries->at(i);

      // Check if new library
      if (loadedLibraries.find(lib) == loadedLibraries.end()) {

        if (path.size() && QFile::exists(QString(path.c_str()) + "/" + lib.c_str()))
          loader.loadLibrary(path + "/" + lib);
        else
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
      if (path.size() && QFile::exists(QString(path.c_str()) + "/" + pos->c_str()))
        loader.unloadLibrary(path + "/" + *pos);
      else
        loader.unloadLibrary(*pos);
    }

    // Sync libraries with loaded libraries (removes duplicates)
    libraries->clear();
    for (set<string>::iterator pos = loadedLibraries.begin(); pos != loadedLibraries.end(); ++pos)
      libraries->push_back(*pos);
  }
}

void Workflow::interfaceChangedHandler(capputils::ObservableClass* sender, int /*eventId*/) {

  // Find the tool connection that corresponds to the sender
  Node* node = dynamic_cast<Node*>(sender);
  if (!node)
    return;

  ToolItem* toolItem = getToolItem();
  if (!toolItem)
    return;

  boost::shared_ptr<ToolConnection> connection;
  if (isInputNode(node->shared_from_this())) {
    connection = toolItem->getConnection(node->getUuid(), ToolConnection::Input);
  } else {
    connection = toolItem->getConnection(node->getUuid(), ToolConnection::Output);
  }

  boost::shared_ptr<WorkflowElement> element = boost::dynamic_pointer_cast<WorkflowElement>(node->getModule());
  if (connection && element) {
    connection->setLabel(element->getLabel().c_str());
  }
}

void Workflow::removeNode(boost::shared_ptr<Node> node) {
//  cout << "Deleting module: " << item->getLabel() << endl;
  Logbook& dlog = *getLogbook();

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
  for (unsigned i = 0; i < _Nodes->size(); ++i) {
    if (_Nodes->at(i) == node) {
      _Nodes->erase(_Nodes->begin() + i);
    }
  }
}

//bool Workflow::getToolConnectionId(boost::shared_ptr<const Node> node, const std::string& propertyName, unsigned& id) const {
//  assert(node);
//  boost::shared_ptr<ReflectableClass> object = node->getModule();
//  assert(object);
//
//  if (object->getPropertyIndex(id, propertyName))
//    return true;
//
//  boost::shared_ptr<const Workflow> workflow = boost::dynamic_pointer_cast<const Workflow>(node);
//  if (workflow) {
//    const std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow->interfaceNodes;
//    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
//      if (interfaceNodes[i].lock()->getUuid() == propertyName) {
//        id = object->getProperties().size() + i;
//        return true;
//      }
//    }
//  }
//
//  return false;
//}

//boost::enable_shared_from_this<Workflow>::
void Workflow::removeEdge(boost::shared_ptr<Edge> edge) {
  vector<boost::shared_ptr<Edge> >& edges = *getEdges();
  for (size_t i = 0; i < edges.size(); ++i) {
    if (edges[i] == edge) {
      edges.erase(edges.begin() + i);
    }
  }

  for (size_t iEdge = 0, pos = 0; iEdge < edges.size(); ++iEdge) {
    if (edges[iEdge]->getInputNode() == edge->getInputNode() &&
        edges[iEdge]->getInputProperty() == edge->getInputProperty())
    {
      edges[iEdge]->setInputPosition(pos++);
    }
  }
}

bool Workflow::areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const {
  assert(output);
  assert(input);

  boost::shared_ptr<const Node> outputNode = getNode(output->parent);
  boost::shared_ptr<const Node> inputNode = getNode(input->parent);

  if (inputNode && outputNode) {
    PropertyReference outRef(shared_from_this(), outputNode->getUuid(), output->id);
    PropertyReference inRef(shared_from_this(), inputNode->getUuid(), input->id);

    return Edge::areCompatible(outRef.getProperty(), inRef.getProperty());
  }

  return false;
}

bool Workflow::isInterfaceNode(boost::shared_ptr<const Node> node) const {
  if (!node)
    return false;

  boost::shared_ptr<ReflectableClass> module = node->getModule();

  if (!module)
    return false;

  return module->getAttribute<InterfaceAttribute>();
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

void Workflow::getDependentNodes(boost::shared_ptr<Node> node, std::vector<boost::shared_ptr<Node> >& dependendNodes, bool includeParentDependencies) {
  bool isWorkflow = boost::dynamic_pointer_cast<Workflow>(node);

  // If input node see to which node of the parent workflow this node is connected
  if (isInputNode(node)) {
    if (!includeParentDependencies)
      return;
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
        PropertyReference ref(shared_from_this(), node->getUuid(), edge->getInputProperty());
        if (!ref.getProperty()->getAttribute<NoParameterAttribute>() || isWorkflow)
          dependendNodes.push_back(getNode(edge->getOutputNode()));
      }
    }

    vector<boost::shared_ptr<GlobalEdge> >& gedges = *getGlobalEdges();
    for (unsigned i = 0; i < gedges.size(); ++i) {
      boost::shared_ptr<GlobalEdge> gedge = gedges[i];
      if (gedge->getInputNode() == node->getUuid()) {
        PropertyReference ref(shared_from_this(), node->getUuid(), gedge->getInputProperty());
        if (!ref.getProperty()->getAttribute<NoParameterAttribute>() || isWorkflow)
          dependendNodes.push_back(getNode(gedge->getOutputNode()));
      }
    }
  }
}

bool Workflow::isDependentProperty(boost::shared_ptr<const Node> node, const std::string& propertyName) const {
  if (isInputNode(node)) {
    boost::shared_ptr<Workflow> workflow = getWorkflow().lock();
    const bool isCollection = boost::dynamic_pointer_cast<const CollectionElement>(node->getModule());
    if (((!isCollection && propertyName == "Value") || (isCollection && propertyName == "Values"))
        && workflow)
    {
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

  // Is there a property with a ToEnumerable attribute that points to this property?
  if (isOutputNode(node) && collection && collection->getCalculateCombinations() && propertyName == "Values")
    return true;

  return false;
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
  return getNode(object.get());
}

boost::shared_ptr<Node> Workflow::getNode(boost::shared_ptr<capputils::reflection::ReflectableClass> object, unsigned& pos) {
  return getNode(object.get(), pos);
}

boost::shared_ptr<Node> Workflow::getNode(const capputils::reflection::ReflectableClass* object) {
  unsigned pos;
  return getNode(object, pos);
}

boost::shared_ptr<Node> Workflow::getNode(const capputils::reflection::ReflectableClass* object, unsigned& pos) {
  boost::shared_ptr<Node> node;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getModule().get() == object) {
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

boost::shared_ptr<Node> Workflow::getNodeByLabel(const std::string& label) const {
  boost::shared_ptr<Node> node;

  for(unsigned pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    WorkflowElement* element = dynamic_cast<WorkflowElement*>(node->getModule().get());
    if (element && element->getLabel() == label)
      return node;

    WorkflowInterface* interface = dynamic_cast<WorkflowInterface*>(node->getModule().get());
    if (interface && interface->getLabel() == label)
      return node;
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

boost::shared_ptr<GlobalProperty> Workflow::getGlobalProperty(const PropertyReference& reference) {
  boost::shared_ptr<GlobalProperty> gprop;

  for(unsigned pos = 0; pos < _GlobalProperties->size(); ++pos) {
    gprop = _GlobalProperties->at(pos);
    if (gprop->getModuleUuid() == reference.getNodeId() && gprop->getPropertyId() == reference.getPropertyId()) {
      return gprop;
    }
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

void Workflow::resetInputs() {
  std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i].lock()->getModule());
    if (collection && isInputNode(interfaceNodes[i].lock()))
      collection->resetCombinations();
  }
}

void Workflow::incrementInputs() {
  std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i].lock()->getModule());
    if (collection && isInputNode(interfaceNodes[i].lock()))
      collection->advanceCombinations();
  }
}

void Workflow::decrementInputs() {
  std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = getInterfaceNodes();
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<CollectionElement> collection = boost::dynamic_pointer_cast<CollectionElement>(interfaceNodes[i].lock()->getModule());
    if (collection && isInputNode(interfaceNodes[i].lock()))
      collection->regressCombinations();
  }
}

}

}
