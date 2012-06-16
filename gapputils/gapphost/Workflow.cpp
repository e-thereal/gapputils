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

#include <gapputils/CombinerInterface.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>
#include <gapputils/LabelAttribute.h>
#include <gapputils/InterfaceAttribute.h>

#include <boost/filesystem.hpp>
#include <boost/units/detail/utility.hpp>

#include <set>
#include <map>
#include <iomanip>

#include "PropertyGridDelegate.h"
#include "CustomToolItemAttribute.h"
#include "InputsItem.h"
#include "OutputsItem.h"
#include "CableItem.h"
#include "Workbench.h"
#include "WorkflowItem.h"
#include "PropertyReference.h"
#include "MakeGlobalDialog.h"
#include "PopUpList.h"
#include "XslTransformation.h"

#include "DataModel.h"
#include "MainWindow.h"

using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;
using namespace std;

namespace gapputils {

using namespace attributes;

namespace workflow {

int Workflow::librariesId;
int Workflow::interfaceId;

BeginPropertyDefinitions(Workflow)

// Libraries must be the first property since libraries must be loaded before all other modules
DefineProperty(Libraries, Enumerable<vector<std::string>*, false>(), Observe(librariesId = PROPERTY_ID))

// Same is true for interfaces, since interfaces are used to build libraries which must be loaded before
// all other modules
ReflectableProperty(Interface, Observe(interfaceId = PROPERTY_ID), TimeStamp(PROPERTY_ID))

// Add Properties of node after libraries (module could be an object of a class of one of the libraries)
ReflectableBase(Node)

DefineProperty(Edges, Enumerable<vector<Edge*>*, true>())
DefineProperty(Nodes, Enumerable<vector<Node*>*, true>())
DefineProperty(GlobalProperties, Enumerable<vector<GlobalProperty*>*, true>())
DefineProperty(GlobalEdges, Enumerable<vector<GlobalEdge*>*, true>())
DefineProperty(InputsPosition)
DefineProperty(OutputsPosition)
DefineProperty(ViewportScale)
DefineProperty(ViewportPosition)
DefineProperty(InputChecksums)

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
 : _InputsPosition(0), _OutputsPosition(0), _ViewportScale(1.0), ownWidget(true), hasIONodes(false),
   processingCombination(false), dryrun(false), worker(0), progressNode(0)
{
  _Libraries = new vector<std::string>();
  _Edges = new vector<Edge*>();
  _Nodes = new vector<Node*>();
  _GlobalProperties = new vector<GlobalProperty*>();
  _GlobalEdges = new vector<GlobalEdge*>();

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

  // Context Menu
  makeGlobal = new QAction("Make Global", propertyGrid);
  removeGlobal = new QAction("Remove Global", propertyGrid);
  connectToGlobal = new QAction("Connect", propertyGrid);
  disconnectFromGlobal = new QAction("Disconnect", propertyGrid);
  connect(makeGlobal, SIGNAL(triggered()), this, SLOT(makePropertyGlobal()));
  connect(removeGlobal, SIGNAL(triggered()), this, SLOT(removePropertyFromGlobal()));
  connect(connectToGlobal, SIGNAL(triggered()), this, SLOT(connectProperty()));
  connect(disconnectFromGlobal, SIGNAL(triggered()), this, SLOT(disconnectProperty()));

  propertyGrid->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(propertyGrid, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
  connect(propertyGrid, SIGNAL(clicked(const QModelIndex&)), this, SLOT(gridClicked(const QModelIndex&)));
  QSplitter* gridSplitter = new QSplitter(Qt::Vertical);
  gridSplitter->addWidget(propertyGrid);

  QWidget* infoWidget = new QWidget();
  infoLayout = new QFormLayout();
  infoWidget->setLayout(infoLayout);
  gridSplitter->addWidget(infoWidget);

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(workbench);
  splitter->addWidget(gridSplitter);
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

#define TRACE std::cout << __LINE__ << std::endl;

Workflow::~Workflow() {
  const std::string className = (getModule() ? getModule()->getClassName() : "none");
  const std::string uuid = getUuid();
//  std::cout << "[Info] Start deleting " << className << " (" << uuid << ")" << std::endl;
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

  inputsNode.setModule(0);
  outputsNode.setModule(0);

  // Don't delete module before setting it to zero
  // The module property is observed and reflectable. Thus, when resetting
  // the module, the event listener is disconnected from the old module.
  // This will cause the application to crash when the module has already been
  // deleted.
  ReflectableClass* module = getModule();
  setModule(0);
  if (module)
    delete module;

  // Unload interface library
  if (getInterface()) {
    loader.freeLibrary(getLibraryName());
  }

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
  interfaceNodes.push_back(node);

  ReflectableClass* object = node->getModule();
  assert(object);

  IClassProperty* prop = object->findProperty("Value");
  if (!prop)
    return;

  if (prop->getAttribute<InputAttribute>()) {
    ToolItem* item = getToolItem();
    if (!item) {
      std::cout << "[Info] Workflow does not have a ToolItem" << std::endl;
    } else {
      item->addConnection(QString(object->getProperty("Label").c_str()), interfaceNodes.size() + getModule()->getProperties().size() - 1, ToolConnection::Output);
//      std::cout << "[Info] New connection added with id " << interfaceNodes.size() + getModule()->getProperties().size() - 1 << std::endl;
    }
  }
  if (prop->getAttribute<OutputAttribute>()) {
    ToolItem* item = getToolItem();
    if (!item) {
      std::cout << "[Info] Workflow does not have a ToolItem" << std::endl;
    } else {
      item->addConnection(QString(object->getProperty("Label").c_str()), interfaceNodes.size() + getModule()->getProperties().size() - 1, ToolConnection::Input);
//      std::cout << "[Info] New connection added with id " << interfaceNodes.size() + getModule()->getProperties().size() - 1 << std::endl;
    }
  }
}
  
void Workflow::removeInterfaceNode(Node* node) {
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
            std::cout << "[Info] Workflow does not have a ToolItem" << std::endl;
          } else {
            deletedId = getModule()->getProperties().size() + i;
//            std::cout << "[Info] removedId = " << deletedId << std::endl;
            item->deleteConnection(deletedId, ToolConnection::Output);
          }
        }

        if (prop->getAttribute<OutputAttribute>()) {
          ToolItem* item = getToolItem();
          if (!item) {
            std::cout << "[Info] Workflow does not have a ToolItem" << std::endl;
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

const Node* Workflow::getInterfaceNode(int id) const {
  assert(getModule());
  const int pos = id - getModule()->getProperties().size();
  if (pos >= 0 && (unsigned)pos < interfaceNodes.size())
    return interfaceNodes[pos];
  return 0;
}

QLabel* createTopAlignedLabel(const std::string& text) {
  QLabel* label = new QLabel(text.c_str());
  label->setAlignment(Qt::AlignTop);
  return label;
}

void Workflow::gridClicked(const QModelIndex& index) {
  const QModelIndex& valueIndex = index.sibling(index.row(), 1);
  if (!valueIndex.isValid())
    return;

  QVariant varient = valueIndex.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
   return;

  const PropertyReference& reference = varient.value<PropertyReference>();
  IClassProperty* prop = reference.getProperty();

  while (infoLayout->count()) {
    QLayoutItem *item = infoLayout->takeAt(0);
    if (item) {
      delete item->widget();
      delete item;
    } else {
      break;
    }
  }

  QLabel* description;
  DescriptionAttribute* descAttr = prop->getAttribute<DescriptionAttribute>();
  if (descAttr)
    description = new QLabel((std::string("<b>") + descAttr->getDescription() + "</b>").c_str());
  else
    description = new QLabel((std::string("<b>") + prop->getName() + "</b>").c_str());
  description->setWordWrap(true);
  infoLayout->addRow(description);
  QLabel* typeLabel = new QLabel(boost::units::detail::demangle(prop->getType().name()).c_str());
  typeLabel->setWordWrap(true);
  typeLabel->setMinimumSize(10, 10);
  infoLayout->addRow(createTopAlignedLabel("Type:"), typeLabel);

  // check if global and check if connected and fill actions list accordingly
  GlobalProperty* gprop = getGlobalProperty(reference.getObject(), reference.getProperty());
  if (gprop)
    infoLayout->addRow("Name:", new QLabel(gprop->getName().c_str()));

  GlobalEdge* edge = getGlobalEdge(reference.getObject(), reference.getProperty());
  if (edge)
    infoLayout->addRow("Connection:", new QLabel(edge->getGlobalProperty().c_str()));

  Node* node = getNode(workbench->getCurrentItem());
  if (node) {
    QLabel* uuidLabel = new QLabel(node->getUuid().c_str());
    uuidLabel->setMinimumSize(10, 10);
    infoLayout->addRow("Uuid:", uuidLabel);
  }

  if (reference.getObject()) {
    QLabel* moduleTypeLabel = new QLabel(reference.getObject()->getClassName().c_str());
    moduleTypeLabel->setWordWrap(true);
    moduleTypeLabel->setMinimumSize(10, 10);
    infoLayout->addRow(createTopAlignedLabel("Module:"), moduleTypeLabel);
  }
}

void Workflow::showContextMenu(const QPoint& point) {
  QList<QAction*> actions;
  QModelIndex index = propertyGrid->indexAt(point);
  if (!index.isValid())
    return;

  QVariant varient = index.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
    return;

  const PropertyReference& reference = varient.value<PropertyReference>();

  // check if global and check if connected and fill actions list accordingly
  GlobalProperty* gprop = getGlobalProperty(reference.getObject(), reference.getProperty());
  if (gprop)
    actions.append(removeGlobal);
  else
    actions.append(makeGlobal);
  

  GlobalEdge* edge = getGlobalEdge(reference.getObject(), reference.getProperty());
  if (edge)
    actions.append(disconnectFromGlobal);
  else
    actions.append(connectToGlobal);

  QMenu::exec(actions, propertyGrid->mapToGlobal(point));
}

void Workflow::makePropertyGlobal() {
  host::MakeGlobalDialog dialog(propertyGrid);
  if (dialog.exec() == QDialog::Accepted) {
    QString text = dialog.getText();
    if (text.length()) {
      QModelIndex index = propertyGrid->currentIndex();
      makePropertyGlobal(text.toAscii().data(), index.data(Qt::UserRole).value<PropertyReference>());
    } else {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for a global property!");
    }
  }
}

void Workflow::removePropertyFromGlobal() {
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  GlobalProperty* gprop = getGlobalProperty(reference.getObject(), reference.getProperty());
  if (gprop)
    removeGlobalProperty(gprop);
}

void Workflow::connectProperty() {
  // Get list of compatible global properties.
  // Show list windows.
  // establish connection.
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  host::PopUpList list;
  vector<GlobalProperty*>* globals = getGlobalProperties();
  for (unsigned i = 0; i < globals->size(); ++i) {
    if (Edge::areCompatible(globals->at(i)->getProperty(), reference.getProperty())) {
      list.getList()->addItem(globals->at(i)->getName().c_str());
    } else {
//      GlobalProperty* gprop = globals->at(i);
      //cout << gprop->getName() << " is not compatible." << endl;
      //cout << gprop->getProperty()->getType().name() << " != " << reference.getProperty()->getType().name() << endl;
    }
  }
  if (list.getList()->count() == 0) {
    QMessageBox::information(0, "No Compatible Global Connection", "There are no compatible global connections to connect to.");
    return;
  }

  if (list.exec() == QDialog::Accepted) {
    connectProperty(list.getList()->selectedItems()[0]->text().toAscii().data(), reference);
  }
}

void Workflow::disconnectProperty() {
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  GlobalEdge* edge = getGlobalEdge(reference.getObject(), reference.getProperty());
  if (edge)
    removeGlobalEdge(edge);
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

  unsigned id;
  if (!node->getModule()->getPropertyIndex(id, prop->getPropertyName())) {
    // TODO: Error handling
    cout << "[Warning] Property " << prop->getPropertyName() << " could not be found." << endl;
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
    cout << "no such item, " << __FILE__ << ", " << __LINE__ << endl;
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
  Node* inputNode = getNode(edge->getInputNode());

  GlobalProperty* globalProp = getGlobalProperty(edge->getGlobalProperty());
  globalProp->addEdge(edge);
  if (!edge->activate(getNode(edge->getOutputNode()), inputNode)) {
    // TODO: should not happen but just in case, handle it right
    cout << "Error in line " << __LINE__ << endl;
  }

  QStandardItem* item = getItem(inputNode->getModule(),
      inputNode->getModule()->findProperty(edge->getInputProperty()));
  if (item) {
    QFont font = item->font();
    font.setItalic(true);
    item->setFont(font);
  } else {
    cout << "no such item" << endl;
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
  } else if (node == &inputsNode) {
    item = new ToolItem("Inputs");
    item->setDeletable(false);

    std::set<int> linkedEnumerables;

    if (dynamic_cast<CombinerInterface*>(getModule())) {
      FromEnumerableAttribute* fromEnum = 0;
      for (unsigned i = 0; i < properties.size(); ++i) {
        IClassProperty* prop = properties[i];
        if ((fromEnum = prop->getAttribute<FromEnumerableAttribute>())) {
          linkedEnumerables.insert(fromEnum->getEnumerablePropertyId());
        }
      }
    }

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if (prop->getAttribute<InputAttribute>() && linkedEnumerables.find(i) == linkedEnumerables.end())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Output);
    }
  } else if (node == &outputsNode) {
    item = new ToolItem("Outputs");
    item->setDeletable(false);

    std::set<int> linkedEnumerables;
    ToEnumerableAttribute* toEnum = 0;
    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if ((toEnum = prop->getAttribute<ToEnumerableAttribute>())) {
        linkedEnumerables.insert(toEnum->getEnumerablePropertyId());
      }
    }

    for (unsigned i = 0; i < properties.size(); ++i) {
      IClassProperty* prop = properties[i];
      if (prop->getAttribute<OutputAttribute>() && linkedEnumerables.find(i) == linkedEnumerables.end())
        item->addConnection(getPropertyLabel(prop).c_str(), i, ToolConnection::Input);
    }
  } else if (customToolItem) {
    item = customToolItem->createToolItem(label);
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
    // TODO: Error handling
    cout << "[Warning] Can not find connections for edge " << edge->getInputNode() << " -> " << edge->getOutputNode() << endl;
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

std::string Workflow::getPrefix() {
  if (getInterface())
    return string(".gapphost/") + getInterface()->getName();
  return string(".gapphost/") + getUuid();
}

std::string Workflow::getLibraryName() {
  return getPrefix() + ".so";
}

std::string Workflow::getInterfaceName() {
  return string("Interface_") + replaceAll(getUuid(), "-", "");
}

void Workflow::createAndLoadAdhocModule() {
  using namespace gapputils::host::internal;
  time_t libTime, interfaceTime;
  gapputils::host::DataModel& model = gapputils::host::DataModel::getInstance();
  gapputils::host::BuilderSettings& builderSettings = *model.getBuilderSettings();
  gapputils::host::XsltSettings& xsltSettings = *model.getXsltSettings();

  if (!boost::filesystem::exists(getLibraryName().c_str()) ||
      ((libTime = boost::filesystem::last_write_time(getLibraryName().c_str())) < (interfaceTime = getTime(interfaceId))))
  {
    //getInterface()->setName(getInterfaceName());
    string prefix = getPrefix();
    Xmlizer::ToXml(prefix + ".xml", *getInterface());

    capputils::Executer xslt;
    xslt.getCommand() << xsltSettings.getCommandName() << " "
                      << xsltSettings.getInputSwitch() << "\"" << prefix << ".xml\" "
                      << xsltSettings.getOutputSwitch() << "\"" << prefix << ".cpp\" "
                      << xsltSettings.getXsltSwitch() << "\"" << model.getConfigurationDirectory() << "/";
    if (getInterface()->getIsCombinerInterface())
      xslt.getCommand() << xsltSettings.getCombinerInterfaceStyleSheetName() << "\"";
    else
      xslt.getCommand() << xsltSettings.getStandardInterfaceStyleSheetName() << "\"";
    cout << "[Info] " << xslt.getCommandString() << endl;
    if (xslt.execute()) {
      cout << "[Error] Xsl transformation failed: " << xslt.getOutput() << endl;
      return;
    }

    capputils::Executer build;
    build.getCommand() << builderSettings.getCompilerName();

    std::vector<std::string>& flags = *builderSettings.getCompilerFlags();
    for (unsigned i = 0; i < flags.size(); ++i)
      build.getCommand() << " " << flags[i];

    std::vector<std::string>& includeDirs = *builderSettings.getIncludeDirectories();
    for (unsigned i = 0; i < includeDirs.size(); ++i)
      build.getCommand() << " " << builderSettings.getIncludeSwitch() << "\"" << includeDirs[i] << "\"";

    build.getCommand() << " " << prefix << ".cpp " << builderSettings.getOutputSwitch() << getLibraryName();

    cout << "[Info] " << build.getCommandString() << endl;
    if (build.execute()) {
      // TODO: report error
      cout << "[Error] Compilation failed: " << build.getOutput() << endl;
      return;
    }
  }

  LibraryLoader::getInstance().loadLibrary(getLibraryName());
  ReflectableClass* module = getModule();
  setModule(ReflectableClassFactory::getInstance().newInstance(string("gapputils::host::internal::") + getInterface()->getName()));
  inputsNode.setModule(getModule());
  outputsNode.setModule(getModule());
  if (module)
    delete module;
}

void Workflow::resume() {
  map<string, Workflow*>* workflowMap = host::DataModel::getInstance().getWorkflowMap().get();
  //assert(workflowMap->find(getUuid()) == workflowMap->end());
  if (workflowMap->find(getUuid()) == workflowMap->end())
    workflowMap->insert(pair<string, Workflow*>(getUuid(), this));

  if (!hasIONodes) {
    hasIONodes = true;
    inputsNode.setModule(getModule());
    inputsNode.setX(getInputsPosition()[0]);
    inputsNode.setY(getInputsPosition()[1]);
    inputsNode.setWorkflow(this);
    outputsNode.setModule(getModule());
    outputsNode.setX(getOutputsPosition()[0]);
    outputsNode.setY(getOutputsPosition()[1]);
    outputsNode.setWorkflow(this);

    newItem(&inputsNode);
    newItem(&outputsNode);
  }

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
      cout << "[Info] Edge has been removed from the model." << endl;
    }
  }

  for (unsigned i = 0; i < globals->size(); ++i) {
    if (!activateGlobalProperty(globals->at(i))) {
      cout << "[Info] Removing global property." << endl;
      removeGlobalProperty(globals->at(i));
      --i; // because there is now one less gprob
    }
  }

  for (unsigned i = 0; i < gedges->size(); ++i)
    activateGlobalEdge(gedges->at(i));

  for (unsigned i = 0; i < nodes->size(); ++i)
    nodes->at(i)->resumeExpressions();
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
  } else if (eventId == interfaceId && getInterface()) {
    createAndLoadAdhocModule();
  }
}

void Workflow::updateInterfaceTimeStamp() {
  setCurrentTime(interfaceId);
}

void Workflow::itemSelected(ToolItem* item) {
  Node* node = getNode(item);
  if (node) {
    propertyGrid->setModel(node->getModel());
    propertyGrid->expandAll();
  } else {
    propertyGrid->setModel(0);
  }
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
  //cout << "Deleting module: " << item->getLabel() << endl;

  unsigned i = 0;
  Node* node = getNode(item, i);

  if (!node) {
    cout << "[Error] Node not found! " << __FILE__ << ", " << __LINE__ << endl;
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
      if (!outputNode->getModule()->getPropertyIndex(outputId, "Value"))
        outputNode = 0;
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
      if (!inputNode->getModule()->getPropertyIndex(inputId, "Value"))
        inputNode = 0;
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
    if (edge->getInputNode() == node->getUuid()) {
      buildStack(getNode(edge->getOutputNode()));
    }
  }
}

void Workflow::updateCurrentModule() {
  //return;

  //cout << "[" << QThread::currentThreadId() << "] " << "Update selected module" << endl;
  // build stack
  Node* node = getNode(workbench->getCurrentItem());
  if (!node)
    return;

  if (host::DataModel::getInstance().getMainWorkflow() == this) {
    ReflectableClass* module = getModule();
    CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(module);
    std::vector<checksum_type> checksums;
    if (module) {
      const std::vector<IClassProperty*>& properties = module->getProperties();
      for (unsigned i = 0; i < properties.size(); ++i) {
//        std::cout << properties[i]->getName() << std::endl;
        if (properties[i]->getAttribute<InputAttribute>() &&
            (!combiner || !properties[i]->getAttribute<FromEnumerableAttribute>()))
        {
          int cs = getChecksum(properties[i], *module);
          checksums.push_back(cs);
        }
      }
    }
    updateChecksum(checksums, node);
  } else {
    updateChecksum(getInputChecksums(), node);
  }

  buildStack(node);
  processStack();
}

Workflow* Workflow::getCurrentWorkflow() {
  Node* node = getNode(workbench->getCurrentItem());
  if (node == &inputsNode || node == &outputsNode)
    return this;
  return dynamic_cast<Workflow*>(node);
}

void Workflow::updateOutputs(bool updateNodes) {
  // if multiple interface
  //  - clear multiple outputs
  //  - reset combinations iterator
  if (host::DataModel::getInstance().getMainWorkflow() == this) {
    ReflectableClass* module = getModule();
    CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(module);
    std::vector<checksum_type> checksums;
    if (module) {
      const std::vector<IClassProperty*>& properties = module->getProperties();
      for (unsigned i = 0; i < properties.size(); ++i) {
//        std::cout << properties[i]->getName() << std::endl;
        if (properties[i]->getAttribute<InputAttribute>() &&
            (!combiner || !properties[i]->getAttribute<FromEnumerableAttribute>()))
        {
          int cs = getChecksum(properties[i], *module);
          checksums.push_back(cs);
        }
      }
    }
    updateChecksum(checksums);
  } else {
    updateChecksum(getInputChecksums());
  }

  if (!updateNodes && isUpToDate()) {
    processStack();         ///< This emits the finished signal
    return;
  }

  this->updateNodes();
}

void Workflow::abortUpdate() {
  // Set the abort flag to true of the progress monitor of the current module
  processingCombination = false;
  while (!nodeStack.empty()) {
    Node* node = nodeStack.top();
    nodeStack.pop();

    processedStack.push(node);
  }
  worker->abort();
}

void Workflow::updateNodes() {
  CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(getModule());
  if (combiner) {
    if (combiner->resetCombinations())
      processingCombination = true;
    else {
      processStack();
      return;
    }
  }
  buildStack(&outputsNode);
  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    // TODO: check if this builds a correct stack
    if (interfaceNodes[i]->getModule()->findProperty("Value")->getAttribute<InputAttribute>())
      buildStack(interfaceNodes[i]);
  }

  processStack();
}

void Workflow::processStack() {
  CombinerInterface* combiner = dynamic_cast<CombinerInterface*>(getModule());

  while (!nodeStack.empty()) {
    Node* node = nodeStack.top();
    nodeStack.pop();

    processedStack.push(node);

    // TODO: Single update in a combiner interface is not handled correctly.
    //       It is unclear how to determine if a module is up-to-date or not.

    // Update the node, if it needs update or if it is the last one
    if (nodeStack.empty() || !node->isUpToDate() || processingCombination || combiner)
    {
      node->getToolItem()->setProgress(-2);
      Workflow* workflow = dynamic_cast<Workflow*>(node);
      if (workflow) {
        connect(workflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(finalizeModuleUpdate(workflow::Node*)));
        workflow->updateNodes();
      } else {
        Q_EMIT processModule(node, nodeStack.empty());
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

  if (combiner && processingCombination) {
    combiner->appendResults();
    if (combiner->advanceCombinations()) {
      ToolItem* item = getToolItem();
      if (item)
        item->setProgress(combiner->getProgress());
      buildStack(&outputsNode);
      processStack();
      return;       // return here. otherwise update finished is emitted.
    }
    processingCombination = false;
  }

  Q_EMIT updateFinished(this);
}

void Workflow::finalizeModuleUpdate(Node* node) {
  progressNode = 0;
  host::DataModel::getInstance().getMainWindow()->statusBar()->showMessage("Ready.");

  Workflow* workflow = dynamic_cast<Workflow*>(node);
  if (workflow) {
    disconnect(workflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(finalizeModuleUpdate(workflow::Node*)));
  } else {
    node->writeResults();
  }
  node->getToolItem()->setProgress(100);
  if (node == &outputsNode) {
    // The output checksum of current workflow to its input checksum
    setOutputChecksum(getInputChecksum());
  } else {
    // Set output checksum of node to its input checksum
    node->setOutputChecksum(node->getInputChecksum());
  }

  // processStack (will emit updateFinished signal)
  processStack();
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

void Workflow::showProgress(Node* node, double progress, bool updateNode) {
  node->getToolItem()->setProgress(progress);
  if (updateNode) {
    node->writeResults();
  }

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

void Workflow::update(IProgressMonitor*, bool) {
}

void Workflow::writeResults() {
}

void Workflow::updateChecksum(const std::vector<checksum_type>& inputChecksums) {
  updateChecksum(inputChecksums, &outputsNode);
}

void Workflow::updateChecksum(const std::vector<checksum_type>& inputChecksums, Node* node) {
  setInputChecksums(inputChecksums);
  buildStack(node);
  while (!nodeStack.empty()) {
    Node* node = nodeStack.top();
    nodeStack.pop();

    if (node == &inputsNode) {
      node->updateChecksum(inputChecksums);
    } else {

      // Collect checksums of all direct inputs
      std::vector<checksum_type> checksums;
      vector<Edge*>* edges = getEdges();
      std::set<std::string> connectedProperties;
      for (int j = (int)edges->size() - 1; j >= 0; --j) {
        Edge* edge = edges->at(j);
        if (edge->getInputNode() == node->getUuid()) {
          Node* outputNode = getNode(edge->getOutputNode());
          if (outputNode) {
            checksums.push_back(outputNode->getInputChecksum());
            connectedProperties.insert(edge->getInputProperty());
          }
        }
      }

      // Collect checksums of all unconnected input properties
      if (node != &outputsNode) {
        ReflectableClass* module = node->getModule();
        if (module) {
          const std::vector<IClassProperty*>& properties = module->getProperties();
          for (unsigned i = 0; i < properties.size(); ++i) {
            // Ignore it if it is not an input or a connected input
            if (!properties[i]->getAttribute<InputAttribute>() ||
                properties[i]->getAttribute<FromEnumerableAttribute>() ||
                connectedProperties.find(properties[i]->getName()) != connectedProperties.end())
            {
              continue;
            }
            std::cout << "[Checksum Module] Found unconnected input property: " << properties[i]->getName() << std::endl;

            int cs = Node::getChecksum(properties[i], *module);
            checksums.push_back(cs);
          }
        }
      }

      node->updateChecksum(checksums);
    }

    if (node == &outputsNode) {
      setInputChecksum(node->getInputChecksum());
    }
  }
}

void Workflow::delegateDeleteCalled(workflow::Workflow* workflow) {
  Q_EMIT deleteCalled(workflow);
}

void Workflow::load(const string& filename) {
  // Delete all current nodes, edges
  // Load model data from xml file (only selected properties)
  // Replace workflows's uuid with new uuid (also in interface name
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
  Xmlizer::GetPropertyFromXml(*this, findProperty("Interface"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("Nodes"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("Edges"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("GlobalProperties"), filename);
  Xmlizer::GetPropertyFromXml(*this, findProperty("GlobalEdges"), filename);

  InterfaceDescription* interface = getInterface().get();
  if (interface)
    interface->setName(getInterfaceName());

  gapputils::host::DataModel::getInstance().getMainWindow()->reload();
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
      if (toolItem->isDeletable() && !dynamic_cast<Workflow*>(node)) {
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

    // Not implemented unless the ID change is applied to all
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
      cout << "[Info] Edge has been removed from the model." << endl;
    }
  }

  edges.clear(); // avoid double free memory
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

Node* Workflow::getNode(capputils::reflection::ReflectableClass* object) {
  unsigned pos;
  if (object == inputsNode.getModule())
    return &inputsNode;
  else if (object == outputsNode.getModule())
    return &outputsNode;
  else
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

  if (inputsNode.getUuid().compare(uuid) == 0)
    return &inputsNode;
  else if (outputsNode.getUuid().compare(uuid) == 0)
    return &outputsNode;

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

void Workflow::updateCache() {
}

bool Workflow::restoreFromCache() {
  return false;
}

PropertyReference* Workflow::getPropertyReference(const std::string& propertyName) {
  PropertyReference* ref = Node::getPropertyReference(propertyName);

  if (ref)
    return ref;

  for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
    if (interfaceNodes[i]->getUuid() == propertyName) {
      ref = interfaceNodes[i]->getPropertyReference("Value");
      ref->setNode(this);
      return ref;
    }
  }

  return 0;
}

}

}

