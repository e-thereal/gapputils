/*
 * Workflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#define BOOST_FILESYSTEM_VERSION 2

#include "Workflow.h"

#include <qtreeview.h>
#include <qsplitter.h>
#include <qaction.h>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qformlayout.h>
#include <qlabel.h>
#include <qtextedit.h>
#include <qclipboard.h>
#include <qstatusbar.h>
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
#include <capputils/Verifier.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Executer.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/DescriptionAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>
#include <capputils/Logbook.h>

#include <gapputils/CombinerInterface.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/InterfaceAttribute.h>
#include <gapputils/CollectionElement.h>

#include <boost/filesystem.hpp>
#include <boost/units/detail/utility.hpp>

#include <set>
#include <map>
#include <iomanip>

#include "PropertyGridDelegate.h"
#include "CableItem.h"
#include "Workbench.h"
#include "WorkflowItem.h"
#include "PropertyReference.h"
#include "MakeGlobalDialog.h"
#include "PopUpList.h"

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
DefineProperty(Libraries, Enumerable<vector<std::string>*, false>(), Observe(librariesId = PROPERTY_ID))

// Add Properties of node after libraries (module could be an object of a class of one of the libraries)
ReflectableBase(Node)

DefineProperty(Edges, Enumerable<vector<Edge*>*, true>())
DefineProperty(Nodes, Enumerable<vector<Node*>*, true>())
DefineProperty(GlobalProperties, Enumerable<vector<GlobalProperty*>*, true>())
DefineProperty(GlobalEdges, Enumerable<vector<GlobalEdge*>*, true>())
DefineProperty(ViewportScale)
DefineProperty(ViewportPosition)
DefineProperty(Logbook, Volatile())

EndPropertyDefinitions

/**

  TODOs:

  - Copy action. Creates an XML Workflow containing the to be copied nodes + non dangling edges

  - Insert action. Inserts nodes and edges from on an XML Workflow. Renames all uuids.
    Parses for UUIDs. If replacement exists, apply replacement, otherwise create new UUID.

  - Code snippets: Select nodes and drag and drop them to the code snippets window underneath the toolbox.
    Dropping opens a Dialogbox where the user can enter the name of the snippet. (renaming later)
    Drag a snippet from the snippets window and drop it into the workbench creates according nodes.
    Code snippets act like named copy and paste

  - Make Workflow Refactory: Puts selected nodes into a subworkflow.
    Determines inputs and outputs automatically (all edges that are connected to non-subworkflow nodes create an input or output property)

 */

Workflow::Workflow()
 : _ViewportScale(1.0), _Logbook(new Logbook(&host::LogbookModel::GetInstance())),
   ownWidget(true), progressNode(0), workflowUpdater(new host::WorkflowUpdater())
{
  _Libraries = new vector<std::string>();
  _Edges = new vector<Edge*>();
  _Nodes = new vector<Node*>();
  _GlobalProperties = new vector<GlobalProperty*>();
  _GlobalEdges = new vector<GlobalEdge*>();

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

  connect(workflowUpdater.get(), SIGNAL(progressed(workflow::Node*, double)), this, SLOT(showProgress(workflow::Node*, double)));
  connect(workflowUpdater.get(), SIGNAL(updateFinished()), this, SLOT(workflowUpdateFinished()));

  this->Changed.connect(EventHandler<Workflow>(this, &Workflow::changedHandler));
}

#define TRACE std::cout << __LINE__ << std::endl;

Workflow::~Workflow() {
  const std::string className = (getModule() ? getModule()->getClassName() : "none");
  const std::string uuid = getUuid();
//  std::cout << "[Info] Start deleting " << className << " (" << uuid << ")" << std::endl;
  Q_EMIT deleteCalled(this);

  LibraryLoader& loader = LibraryLoader::getInstance();

  if (ownWidget) {
    delete widget;
  }

  for (unsigned i = 0; i < _Edges->size(); ++i)
    delete _Edges->at(i);
  delete _Edges;

  for (unsigned i = 0; i < _GlobalEdges->size(); ++i)
    delete _GlobalEdges->at(i);
  delete _GlobalEdges;

  // Delete expressions first because expressions link to nodes and
  // disconnect when deleted. Thus, the node must not be deleted before
  // deleting the expression
  for (unsigned i = 0; i < _Nodes->size(); ++i)
    _Nodes->at(i)->getExpressions()->clear();

  for (unsigned i = 0; i < _Nodes->size(); ++i)
    delete _Nodes->at(i);
  delete _Nodes;

  for (unsigned i = 0; i < _GlobalProperties->size(); ++i)
    delete _GlobalProperties->at(i);
  delete _GlobalProperties;

  // Don't delete module before setting it to zero
  // The module property is observed and reflectable. Thus, when resetting
  // the module, the event listener is disconnected from the old module.
  // This will cause the application to crash when the module has already been
  // deleted.
  ReflectableClass* module = getModule();
  setModule(0);
  if (module)
    delete module;

  // Unload libraries
  for (unsigned i = 0; i < _Libraries->size(); ++i) {
//    std::cout << _Libraries->at(i) << std::endl;
    loader.freeLibrary(_Libraries->at(i));
  }

  delete _Libraries;

  map<string, Workflow*>* workflowMap = host::DataModel::getInstance().getWorkflowMap().get();
  //assert(workflowMap->find(getUuid()) != workflowMap->end());
  if (workflowMap && workflowMap->find(getUuid()) != workflowMap->end())
    workflowMap->erase(getUuid());

//  std::cout << "[Info] Finished deleting " << className << " (" << uuid << ")" << std::endl;
}

void Workflow::addInterfaceNode(Node* node) {
  Logbook& dlog = *getLogbook();
  interfaceNodes.push_back(node);

  ReflectableClass* object = node->getModule();
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
  
void Workflow::removeInterfaceNode(Node* node) {
  Logbook& dlog = *getLogbook();
//  std::cout << "[Info] removing interface node" << std::endl;

  int deletedId = -1;
  ReflectableClass* object = node->getModule();
  assert(object);

  IClassProperty* prop = object->findProperty("Value");

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i] == node) {
      interfaceNodes.erase(interfaceNodes.begin() + i);
//      std::cout << "[Info] removed idx = " << i << std::endl;

      // delete edges connected to the node/tool connection
      Workflow* parent = getWorkflow();
      if (parent) {
        vector<Edge*>* edges = parent->getEdges();
        for (int j = (int)edges->size() - 1; j >= 0; --j) {
          Edge* edge = edges->at(j);
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
    std::vector<ToolConnection*> outputs;
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
    std::vector<ToolConnection*>& inputs = item->getInputs();

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
  CollectionElement* collection = 0;
  for (unsigned i = 0; i < interfaceNodes.size(); ++i)
    if ((collection = dynamic_cast<CollectionElement*>(interfaceNodes[i]->getModule())) && collection->getCalculateCombinations())
      return true;
  return false;
}

const Node* Workflow::getInterfaceNode(int id) const {
  assert(getModule());
  const int pos = id - getModule()->getProperties().size();
  if (pos >= 0 && (unsigned)pos < interfaceNodes.size())
    return interfaceNodes[pos];
  return 0;
}

std::vector<Node*>& Workflow::getInterfaceNodes() {
  return interfaceNodes;
}


void Workflow::makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference) {
  GlobalProperty* globalProperty = new GlobalProperty();
  Node* node = getNode(propertyReference.getObject());
  globalProperty->setName(name);
  globalProperty->setModuleUuid(node->getUuid());
  globalProperty->setPropertyName(propertyReference.getProperty()->getName());
  getGlobalProperties()->push_back(globalProperty);

  activateGlobalProperty(globalProperty);
}

bool Workflow::activateGlobalProperty(GlobalProperty* prop) {
  Node* node = getNode(prop->getModuleUuid());
  Logbook& dlog = *getLogbook();

  unsigned id;
  if (!node || !node->getModule() || !node->getModule()->getPropertyIndex(id, prop->getPropertyName())) {
    dlog(Severity::Warning) << "Property '" << prop->getPropertyName() << "' could not be found.";
    return false;
  }
  prop->setPropertyId(id);
  prop->setNodePtr(node);

  QStandardItem* item = getItem(prop->getNodePtr()->getModule(),
      prop->getNodePtr()->getModule()->getProperties()[id]);
  if (item) {
    QFont font = item->font();
    font.setUnderline(true);
    item->setFont(font);
  } else {
    dlog(Severity::Warning) << "No such item, " << __FILE__ << ", " << __LINE__;
  }

  return true;
}

void Workflow::connectProperty(const std::string& name, const PropertyReference& propertyReference) {
  // Find global property instance by name
  GlobalProperty* globalProp = getGlobalProperty(name);

  Node* inputNode = getNode(propertyReference.getObject());

  GlobalEdge* edge = new GlobalEdge();
  edge->setOutputNode(globalProp->getModuleUuid());
  edge->setOutputProperty(globalProp->getPropertyName());
  edge->setInputNode(inputNode->getUuid());
  edge->setInputProperty(propertyReference.getProperty()->getName());
  edge->setGlobalProperty(name);

  getGlobalEdges()->push_back(edge);
  activateGlobalEdge(edge);
}

void Workflow::activateGlobalEdge(GlobalEdge* edge) {
  Logbook& dlog = *getLogbook();
  Node* inputNode = getNode(edge->getInputNode());

  GlobalProperty* globalProp = getGlobalProperty(edge->getGlobalProperty());
  globalProp->addEdge(edge);
  if (!edge->activate(getNode(edge->getOutputNode()), inputNode)) {
    dlog(Severity::Error) << "Error in line " << __LINE__;
  }

  QStandardItem* item = getItem(inputNode->getModule(),
      inputNode->getModule()->findProperty(edge->getInputProperty()));
  if (item) {
    QFont font = item->font();
    font.setItalic(true);
    item->setFont(font);
  } else {
    dlog(Severity::Warning) << "no such item: " << __LINE__ << endl;
  }
}

void Workflow::removeGlobalProperty(GlobalProperty* gprop) {
  // remove edges and expressions to that property first
  while(gprop->getEdges()->size()) {
    removeGlobalEdge((GlobalEdge*)gprop->getEdges()->at(0));
  }

  while(gprop->getExpressions()->size()) {
    gprop->getExpressions()->at(0)->disconnect(gprop);
  }

  for (unsigned i = 0; i < _GlobalProperties->size(); ++i) {
    if (_GlobalProperties->at(i) == gprop) {
      _GlobalProperties->erase(_GlobalProperties->begin() + i);
      break;
    }
  }
  
  if (!gprop->getNodePtr() || !gprop->getNodePtr()->getModule()) {
    delete gprop;
    return;
  }

  QStandardItem* item = getItem(gprop->getNodePtr()->getModule(), gprop->getProperty());
  assert(item);

  QFont font = item->font();
  font.setUnderline(false);
  item->setFont(font);

  delete gprop;
}

void Workflow::removeGlobalEdge(GlobalEdge* edge) {
  // TODO: need a better getItem method
  QStandardItem* item = getItem(edge->getInputNodePtr()->getModule(),
    edge->getInputNodePtr()->getModule()->findProperty(edge->getInputProperty()));

  GlobalProperty* gprop = getGlobalProperty(edge->getGlobalProperty());
  assert(gprop);
  gprop->removeEdge(edge);
  for (unsigned i = 0; i < _GlobalEdges->size(); ++i) {
    if (_GlobalEdges->at(i) == edge) {
      _GlobalEdges->erase(_GlobalEdges->begin() + i);
      break;
    }
  }
  delete edge;

  if (item) {
    QFont font = item->font();
    font.setItalic(false);
    item->setFont(font);
  }
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
    workflow->resume();
    connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(delegateDeleteCalled(workflow::Workflow*)));
    connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
    node = workflow;
  } else {
    node = new Node();
    node->setModule(object);
    node->resume();
  }
  node->setWorkflow(this);
  node->setX(x);
  node->setY(y);
  getNodes()->push_back(node);

  if (object->getAttribute<InterfaceAttribute>()) {
    addInterfaceNode(node);
  }

  newItem(node);
}

const std::string& getPropertyLabel(IClassProperty* prop) {
  ShortNameAttribute* shortName = prop->getAttribute<ShortNameAttribute>();
  if (shortName) {
    return shortName->getName();
  }
  return prop->getName();
}

void Workflow::newItem(Node* node) {
  ToolItem* item;
  assert(node->getModule());

  // Get the label
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
      IClassProperty* prop = properties[i];
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

bool Workflow::newCable(Edge* edge) {
  Logbook& dlog = *getLogbook();

//  cout << "Connecting " << edge->getOutputNode() << "." << edge->getOutputProperty()
//       << " with " << edge->getInputNode() << "." << edge->getInputProperty() << "... " << flush;
//
  const string outputNodeUuid = edge->getOutputNode();
  const string inputNodeUuid = edge->getInputNode();

  vector<Node*>* nodes = getNodes();

  Node *outputNode = 0, *inputNode = 0;

  for (unsigned i = 0; i < nodes->size(); ++i) {
    if (nodes->at(i)->getUuid().compare(outputNodeUuid) == 0)
      outputNode = nodes->at(i);
    if (nodes->at(i)->getUuid().compare(inputNodeUuid) == 0)
      inputNode = nodes->at(i);
  }

  ToolConnection *outputConnection = 0, *inputConnection = 0;
  if (outputNode && inputNode) {

    // TODO: try to find the correct propertyId. If the property is not a property
    //       of the module, go through the list of interface nodes and try to find
    //       the property there. PropertyNames of interface nodes are the node ID.
    unsigned outputPropertyId, inputPropertyId;
    if (getToolConnectionId(outputNode, edge->getOutputProperty(), outputPropertyId) &&
        getToolConnectionId(inputNode, edge->getInputProperty(), inputPropertyId))
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

void Workflow::resumeViewport() {
  vector<double> pos = getViewportPosition();
  workbench->setViewScale(getViewportScale());
  workbench->centerOn(pos[0], pos[1]);
  handleViewportChanged();
}

string replaceAll(const string& context, const string& from, const string& to)
{
  string str = context;
  size_t lookHere = 0;
  size_t foundHere;
  while((foundHere = str.find(from, lookHere)) != string::npos)
  {
    str.replace(foundHere, from.size(), to);
        lookHere = foundHere + to.size();
  }
  return str;
}

void Workflow::resume() {
  Logbook& dlog = *getLogbook();

  map<string, Workflow*>* workflowMap = host::DataModel::getInstance().getWorkflowMap().get();
  //assert(workflowMap->find(getUuid()) == workflowMap->end());
  if (workflowMap->find(getUuid()) == workflowMap->end())
    workflowMap->insert(pair<string, Workflow*>(getUuid(), this));

  vector<Node*>* nodes = getNodes();
  vector<Edge*>* edges = getEdges();
  vector<GlobalProperty*>* globals = getGlobalProperties();
  vector<GlobalEdge*>* gedges = getGlobalEdges();

  for (unsigned i = 0; i < nodes->size(); ++i)
    resumeNode(nodes->at(i));

  for (unsigned i = 0; i < edges->size(); ++i) {
    if (!newCable(edges->at(i))) {
      removeEdge(edges->at(i));
      --i;
      dlog() << "Edge has been removed from the model.";
    }
  }

  for (unsigned i = 0; i < globals->size(); ++i) {
    if (!activateGlobalProperty(globals->at(i))) {
      dlog() << "[Info] Removing global property.";
      removeGlobalProperty(globals->at(i));
      --i; // because there are now one less gprob
    }
  }

  for (unsigned i = 0; i < gedges->size(); ++i)
    activateGlobalEdge(gedges->at(i));

  for (unsigned i = 0; i < nodes->size(); ++i)
    nodes->at(i)->resumeExpressions();

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

void Workflow::resumeNode(Node* node) {
  node->setWorkflow(this);
  newItem(node);
  node->resume();
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow) {
    connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(delegateDeleteCalled(workflow::Workflow*)));
    connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
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
  Q_EMIT currentModuleChanged(node);
}

void Workflow::itemChangedHandler(ToolItem* item) {
  Node* node = getNode(item);
  if (node) {
    node->setX(item->x());
    node->setY(item->y());
  }
}

void Workflow::deleteModule(ToolItem* item) {
  //cout << "Deleting module: " << item->getLabel() << endl;
  Logbook& dlog = *getLogbook();

  unsigned i = 0;
  Node* node = getNode(item, i);

  if (!node) {
    dlog(Severity::Error) << "Node not found! " << __FILE__ << ", " << __LINE__;
    return;
  }

  if (node->getModule() && node->getModule()->getAttribute<InterfaceAttribute>()) {
    removeInterfaceNode(node);
  }

  // delete global edges connected to the node
  vector<GlobalEdge*>* gedges = getGlobalEdges();
  for (int j = (int)gedges->size() - 1; j >= 0; --j) {
    GlobalEdge* edge = gedges->at(j);
    if (edge->getInputNode() == node->getUuid()) {
      removeGlobalEdge(edge);
    }
  }

  // delete global properties of the node
  vector<GlobalProperty*>* gprops = getGlobalProperties();
  for (int j = (int)gprops->size() - 1; j >= 0; --j) {
    GlobalProperty* gprop = gprops->at(j);
    if (gprop->getNodePtr() == node) {
      removeGlobalProperty(gprop);
    }
  }

  // delete edges connected to the node
  vector<Edge*>* edges = getEdges();
  for (int j = (int)edges->size() - 1; j >= 0; --j) {
    Edge* edge = edges->at(j);
    if (edge->getInputNode() == node->getUuid() || edge->getOutputNode() == node->getUuid()) {
      removeEdge(edge);
    }
  }

  // remove and delete node
  delete node;
  _Nodes->erase(_Nodes->begin() + i);
}

std::string Workflow::getPropertyName(const Node* node, int connectionId) const {
  const ReflectableClass* object = node->getModule();
  assert(object);
  if (connectionId < 0)
    return "";

  int propertyCount = object->getProperties().size();
  const Workflow* workflow = dynamic_cast<const Workflow*>(node);

  if (connectionId < propertyCount) {
    return object->getProperties()[connectionId]->getName();
  } else if (workflow && (unsigned)connectionId < propertyCount + workflow->interfaceNodes.size()) {
    return workflow->interfaceNodes[connectionId - propertyCount]->getUuid();
  } else {
    return "";
  }
}

bool Workflow::getToolConnectionId(const Node* node, const std::string& propertyName, unsigned& id) const {
  assert(node);
  ReflectableClass* object = node->getModule();
  assert(object);

  if (object->getPropertyIndex(id, propertyName))
    return true;

  const Workflow* workflow = dynamic_cast<const Workflow*>(node);
  if (workflow) {
    const std::vector<Node*>& interfaceNodes = workflow->interfaceNodes;
    for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
      if (interfaceNodes[i]->getUuid() == propertyName) {
        id = object->getProperties().size() + i;
        return true;
      }
    }
  }

  return false;
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
  assert(outputNode && outputNode->getModule() && inputNode && inputNode->getModule());

  Edge* edge = new Edge();

  edge->setOutputNode(outputNode->getUuid());
  edge->setOutputProperty(getPropertyName(outputNode, cable->getInput()->id));
  edge->setInputNode(inputNode->getUuid());
  edge->setInputProperty(getPropertyName(inputNode, cable->getOutput()->id));

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
  unsigned outputId = output->id;
  if (output->id >= (int)outputNode->getModule()->getProperties().size()) {
    // Get interface node and ID of value property
    const Workflow* workflow = dynamic_cast<const Workflow*>(outputNode);
    if (workflow) {
      outputNode = workflow->getInterfaceNode(output->id);
      assert(outputNode->getModule());
      if (dynamic_cast<CollectionElement*>(outputNode->getModule())) {
        if (!outputNode->getModule()->getPropertyIndex(outputId, "Values"))
          outputNode = 0;
      } else {
        if (!outputNode->getModule()->getPropertyIndex(outputId, "Value"))
          outputNode = 0;
      }
    } else {
      outputNode = 0;
    }
  }

  const Node* inputNode = getNode(input->parent);
  unsigned inputId = input->id;
  if (input->id >= (int)inputNode->getModule()->getProperties().size()) {
    // Get interface node and ID of Value property
    const Workflow* workflow = dynamic_cast<const Workflow*>(inputNode);
    if (workflow) {
      inputNode = workflow->getInterfaceNode(input->id);
      assert(inputNode->getModule());
      if (dynamic_cast<CollectionElement*>(outputNode->getModule())) {
        if (!inputNode->getModule()->getPropertyIndex(inputId, "Values"))
          inputNode = 0;
      } else {
        if (!inputNode->getModule()->getPropertyIndex(inputId, "Value"))
          inputNode = 0;
      }
    } else {
      inputNode = 0;
    }
  }

  return Edge::areCompatible(outputNode, outputId, inputNode, inputId);
}

void Workflow::deleteEdge(CableItem* cable) {
  unsigned pos;
  Edge* edge = getEdge(cable, pos);
  if (edge) {
    delete edge;
    _Edges->erase(_Edges->begin() + pos);
  }
}

bool Workflow::isInputNode(const Node* node) const {
  if (!node)
    return false;

  ReflectableClass* module = node->getModule();

  if (!module)
    return false;

  return module->getAttribute<InterfaceAttribute>() && module->findProperty("Value")->getAttribute<OutputAttribute>();
}

bool Workflow::isOutputNode(const Node* node) const {
  if (!node)
    return false;

  ReflectableClass* module = node->getModule();

  if (!module)
    return false;

  return module->getAttribute<InterfaceAttribute>() && module->findProperty("Value")->getAttribute<InputAttribute>();
}

void Workflow::getDependentNodes(Node* node, std::vector<Node*>& dependendNodes) {
  // If input node see to which node of the parent workflow this node is connected
  if (isInputNode(node)) {
    Workflow* workflow = getWorkflow();
    if (workflow) {
      {vector<Edge*>* edges = workflow->getEdges();
      for (unsigned i = 0; i < edges->size(); ++i) {
        Edge* edge = edges->at(i);
        if (edge->getInputNode() == getUuid() && edge->getInputProperty() == node->getUuid())
          dependendNodes.push_back(workflow->getNode(edge->getOutputNode()));
      }}

      {vector<GlobalEdge*>* gedges = workflow->getGlobalEdges();
      for (unsigned i = 0; i < gedges->size(); ++i) {
        GlobalEdge* gedge = gedges->at(i);
        if (gedge->getInputNode() == getUuid() && gedge->getInputProperty() == node->getUuid())
          dependendNodes.push_back(workflow->getNode(gedge->getOutputNode()));
      }}
    }
  } else {
    vector<Edge*>* edges = getEdges();
    for (unsigned i = 0; i < edges->size(); ++i) {
      Edge* edge = edges->at(i);
      if (edge->getInputNode() == node->getUuid()) {
        // TODO: check that the property does not have the NoParameter attribute
        dependendNodes.push_back(getNode(edge->getOutputNode()));
      }
    }

    vector<GlobalEdge*>* gedges = getGlobalEdges();
    for (unsigned i = 0; i < gedges->size(); ++i) {
      GlobalEdge* gedge = gedges->at(i);
      if (gedge->getInputNode() == node->getUuid()) {
        // TODO: check that the property does not have the NoParameter attribute
        dependendNodes.push_back(getNode(gedge->getOutputNode()));
      }
    }
  }
}

bool Workflow::isDependentProperty(const Node* node, const std::string& propertyName) const {
  if (isInputNode(node)) {
    Workflow* workflow = getWorkflow();
    if (workflow && propertyName == "Value") {
      vector<Edge*>* edges = workflow->getEdges();
      for (unsigned i = 0; i < edges->size(); ++i) {
        Edge* edge = edges->at(i);
        if (edge->getInputNode() == getUuid() && edge->getInputProperty() == node->getUuid())
          return true;
      }

      vector<GlobalEdge*>* gedges = workflow->getGlobalEdges();
      for (unsigned i = 0; i < gedges->size(); ++i) {
        GlobalEdge* gedge = gedges->at(i);
        if (gedge->getInputNode() == getUuid() && gedge->getInputProperty() == node->getUuid())
          return true;
      }
    }
  } else {
    vector<Edge*>* edges = getEdges();
    for (unsigned i = 0; i < edges->size(); ++i) {
      Edge* edge = edges->at(i);
      if (edge->getInputNode() == node->getUuid() && edge->getInputProperty() == propertyName)
        return true;
    }

    vector<GlobalEdge*>* gedges = getGlobalEdges();
    for (unsigned i = 0; i < gedges->size(); ++i) {
      GlobalEdge* gedge = gedges->at(i);
      if (gedge->getInputNode() == node->getUuid() && gedge->getInputProperty() == propertyName)
        return true;
    }
  }
  ConstPropertyReference* ref = node->getPropertyReference(propertyName);
  const CollectionElement* collection = dynamic_cast<const CollectionElement*>(node->getModule());
  if (ref && ref->getProperty() && ref->getProperty()->getAttribute<FromEnumerableAttribute>() && collection && collection->getCalculateCombinations())
    return true;

  return false;
}

void Workflow::updateCurrentModule() {
  
  // build stack
  Node* node = getNode(workbench->getCurrentItem());
  if (!node)
    return;

  // update checksums before updating the workflow
  workflowUpdater->update(node);
}

void Workflow::workflowUpdateFinished() {
  for (std::set<Node*>::iterator iter = processedNodes.begin(); iter != processedNodes.end(); ++iter)
    (*iter)->getToolItem()->setProgress(ToolItem::Neutral);
  processedNodes.clear();

  Q_EMIT updateFinished(this);
}

Workflow* Workflow::getCurrentWorkflow() {
  Node* node = getNode(workbench->getCurrentItem());
  return dynamic_cast<Workflow*>(node);
}

Node* Workflow::getCurrentNode() {
  return getNode(workbench->getCurrentItem());
}

void Workflow::updateOutputs() {
  workflowUpdater->update(this);
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

void Workflow::showProgress(Node* node, double progress) {
  node->getToolItem()->setProgress(progress);
  processedNodes.insert(node);

  // TODO: Implement the ETA feature. A timer updates passed time and remaining time.
  //       This function updates estimates total time and time and date when the operation
  //       will have finished.

  if (dynamic_cast<Workflow*>(node))    // no progress for workflows
    return;

  if (node != progressNode) {           // new progress
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

void Workflow::showWorkflow(Workflow* workflow) {
  Q_EMIT showWorkflowRequest(workflow);
}

void Workflow::showWorkflow(ToolItem* item) {
  Node* node = getNode(item);
  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow)
    Q_EMIT showWorkflowRequest(workflow);
}

void Workflow::showModuleDialog(ToolItem* item) {
  Node* node = getNode(item);
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(node->getModule());
  if (element)
    element->show();
}

void Workflow::delegateDeleteCalled(workflow::Workflow* workflow) {
  Q_EMIT deleteCalled(workflow);
}

void Workflow::copySelectedNodesToClipboard() {
  Workflow copyWorkflow;
  std::set<std::string> copied;

  // Temporarily add nodes to the node list for the xmlization.
  // Nodes have to be removed afterwards in order to avoid a double free memory
  std::vector<Node*>* nodes = copyWorkflow.getNodes();
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems()) {
    ToolItem* toolItem = dynamic_cast<ToolItem*>(item);
    if (toolItem) {
      Node* node = getNode(toolItem);
      // TODO: Workflows are not copied unless renewUuid() is fully implemented
      if (!dynamic_cast<Workflow*>(node)) {
        nodes->push_back(node);
        copied.insert(node->getUuid());
      }
    }
  }

  // Add all edges to the workflow where both ends nodes are about to be copied
  std::vector<Edge*>* edges = copyWorkflow.getEdges();
  for (unsigned i = 0; i < getEdges()->size(); ++i) {
    Edge* edge = getEdges()->at(i);
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
  std::vector<Node*>& nodes = *workflow.getNodes();
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

  std::vector<Edge*>& edges = *workflow.getEdges();
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

void Workflow::addNodesFromClipboard() {
  Logbook& dlog = *getLogbook();

  Workflow pasteWorkflow;
  const std::string clipboardText = QApplication::clipboard()->text().toUtf8().data();
  Xmlizer::FromXmlString(pasteWorkflow, clipboardText);

  // Unselect selected items
  Q_FOREACH(QGraphicsItem* item, workbench->scene()->selectedItems())
    item->setSelected(false);

  renewUuids(pasteWorkflow);
  std::vector<Node*>& nodes = *pasteWorkflow.getNodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    getNodes()->push_back(nodes[i]);
    resumeNode(nodes[i]);
    nodes[i]->getToolItem()->setSelected(true);
  }
  nodes.clear(); // avoid double free memory

  std::vector<Edge*>& edges = *pasteWorkflow.getEdges();
  for (unsigned i = 0; i < edges.size(); ++i) {
    getEdges()->push_back(edges[i]);
    if (!newCable(edges[i])) {
      removeEdge(edges[i]);
      dlog() << "Edge has been removed from the model." << endl;
    }
  }

  edges.clear(); // avoid double free memory
}

void Workflow::setUiEnabled(bool enabled) {
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

Node* Workflow::getNode(capputils::reflection::ReflectableClass* object) {
  unsigned pos;
  return getNode(object, pos);
}

Node* Workflow::getNode(capputils::reflection::ReflectableClass* object, unsigned& pos) {
  Node* node = 0;
  for(pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getModule() == object) {
      return node;
    }
  }
  return 0;
}

Node* Workflow::getNode(const std::string& uuid) {
  Node* node = 0;

  for(unsigned pos = 0; pos < _Nodes->size(); ++pos) {
    node = _Nodes->at(pos);
    if (node->getUuid().compare(uuid) == 0) {
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

QStandardItem* Workflow::getItem(capputils::reflection::ReflectableClass* object,
      capputils::reflection::IClassProperty* property)
{
  Node* node = getNode(object);
  if (!node)
    return 0;

  QStandardItemModel* model = node->getModel();
  QStandardItem* root = model->invisibleRootItem();

  for (int i = 0; i < root->rowCount(); ++i) {
    QStandardItem* item = root->child(i, 1);
    QVariant varient = item->data(Qt::UserRole);
    if (varient.canConvert<PropertyReference>()) {
      PropertyReference reference = varient.value<PropertyReference>();
      if (reference.getObject() == object && reference.getProperty() == property)
        return item;
    }
  }

  return 0;
}

GlobalProperty* Workflow::getGlobalProperty(const std::string& name) {
  GlobalProperty* property = 0;

  for(unsigned pos = 0; pos < _GlobalProperties->size(); ++pos) {
    property = _GlobalProperties->at(pos);
    if (property->getName().compare(name) == 0)
      return property;
  }
  return 0;
}

GlobalProperty* Workflow::getGlobalProperty(capputils::reflection::ReflectableClass* object,
  capputils::reflection::IClassProperty* prop)
{
  GlobalProperty* gprop = 0;

  for(unsigned pos = 0; pos < _GlobalProperties->size(); ++pos) {
    gprop = _GlobalProperties->at(pos);
    if (gprop->getNodePtr()->getModule() == object && gprop->getProperty() == prop)
      return gprop;
  }
  return 0;
}

// TODO: re-think how to identify a global edge
GlobalEdge* Workflow::getGlobalEdge(capputils::reflection::ReflectableClass* object,
  capputils::reflection::IClassProperty* prop)
{
  GlobalEdge* edge = 0;
  for(unsigned pos = 0; pos < _GlobalEdges->size(); ++pos) {
    edge = _GlobalEdges->at(pos);
    if (edge->getInputNodePtr()->getModule() == object && edge->getInputProperty().compare(prop->getName()) == 0)
      return edge;
  }
  return 0;
}

PropertyReference* Workflow::getPropertyReference(const std::string& propertyName) {
  PropertyReference* ref = Node::getPropertyReference(propertyName);

  if (ref)
    return ref;

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i]->getUuid() == propertyName) {
      if (dynamic_cast<CollectionElement*>(interfaceNodes[i]->getModule()))
        ref = interfaceNodes[i]->getPropertyReference("Values");
      else
        ref = interfaceNodes[i]->getPropertyReference("Value");
      ref->setNode(this);
      return ref;
    }
  }

  return 0;
}

ConstPropertyReference* Workflow::getPropertyReference(const std::string& propertyName) const {
  ConstPropertyReference* ref = Node::getPropertyReference(propertyName);

  if (ref)
    return ref;

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i]->getUuid() == propertyName) {
      const Node* interfaceNode = interfaceNodes[i];
      if (dynamic_cast<CollectionElement*>(interfaceNode->getModule()))
        ref = interfaceNode->getPropertyReference("Values");
      else
        ref = interfaceNode->getPropertyReference("Value");
      ref->setNode(this);
      return ref;
    }
  }

  return 0;
}

bool Workflow::trySelectNode(const std::string& uuid) {
  Node* node = getNode(uuid);
  if (node && node->getToolItem()) {
    workbench->setExclusivelySelected(node->getToolItem());
    workbench->setCurrentItem(node->getToolItem());
    return true;
  }

  return false;
}

}

}
