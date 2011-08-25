#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <iostream>
#include <qsplitter.h>

#include <capputils/Xmlizer.h>
#include <capputils/LibraryLoader.h>
#include <qtoolbox.h>
#include <qtreewidget.h>

#include "DataModel.h"
#include "Controller.h"
#include "ToolItem.h"
#include "EditInterfaceDialog.h"
#include <capputils/ReflectableClassFactory.h>
#include <qbrush.h>
#include <gapputils/WorkflowElement.h>

#include <iostream>

using namespace std;
using namespace capputils;

namespace gapputils {

using namespace workflow;

namespace host {

QTreeWidgetItem* newCategory(const string& name) {
  QLinearGradient gradient(0, 0, 0, 20);
  gradient.setColorAt(0, Qt::white);
  gradient.setColorAt(1, Qt::lightGray);

  QTreeWidgetItem* item = new QTreeWidgetItem(1);
  item->setBackground(0, gradient);
  item->setText(0, name.c_str());
  item->setTextAlignment(0, Qt::AlignHCenter);

  return item;
}

QTreeWidgetItem* newTool(const string& name, const string& classname) {
  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, name.c_str());
  item->setData(0, Qt::UserRole, QVariant::fromValue(QString(classname.c_str())));
  
  return item;
}

void updateToolBox(QTreeWidget* toolBox) {
  toolBox->setIndentation(5);
  toolBox->setHeaderHidden(true);
  toolBox->setRootIsDecorated(false);
  toolBox->clear();

  reflection::ReflectableClassFactory& factory = reflection::ReflectableClassFactory::getInstance();
  vector<string> classNames = factory.getClassNames();
  sort(classNames.begin(), classNames.end());

  QTreeWidgetItem* item = newCategory("");
  string groupString("");

  for (unsigned i = 0; i < classNames.size(); ++i) {
    string name = classNames[i];
    string currentGroupString;

#ifdef ONLY_WORKFLOWELEMENTS
    reflection::ReflectableClass* object = factory.newInstance(name);
    if (dynamic_cast<WorkflowElement*>(object) == 0) {
      cout << name << " is not a workflow element" << endl;
      delete object;
      continue;
    }
    delete object;
#endif

    int pos = name.find_last_of(":");
    if (pos != (int)string::npos) {
      currentGroupString = name.substr(0, pos-1);
    } else {
      pos = -1;
    }

    if (currentGroupString.compare(groupString)) {
      groupString = currentGroupString;
      item = newCategory(groupString);
      toolBox->addTopLevelItem(item);
    }

    item->addChild(newTool(name.substr(pos+1), name));
  }
  //toolBox->expandAll();
}

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags), libsChanged(false)
{
  DataModel& model = DataModel::getInstance();
  Workflow* workflow = model.getMainWorkflow();

  setWindowTitle("Application Host");
  this->setGeometry(model.getWindowX(), model.getWindowY(), model.getWindowWidth(), model.getWindowHeight());

  newObjectDialog = new NewObjectDialog();

  tabWidget = new QTabWidget();
  tabWidget->setTabsClosable(true);
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
  openWorkflows.push_back(workflow);

  connect(tabWidget, SIGNAL(tabCloseRequested(int)), this, SLOT(closeWorkflow(int)));
  connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
  connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(closeWorkflow(workflow::Workflow*)));

  toolBox = new QTreeWidget();
  toolBox->setDragEnabled(true);
  updateToolBox(toolBox);
  
  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(toolBox);
  splitter->addWidget(tabWidget);
  QList<int> sizes = splitter->sizes();
  sizes[0] = 180;
  sizes[1] = 1100;
  splitter->setSizes(sizes);
  setCentralWidget(splitter);

  fileMenu = menuBar()->addMenu("&File");
  fileMenu->addAction("Load Library", this, SLOT(loadLibrary()), QKeySequence(Qt::CTRL + Qt::Key_L));

  fileMenu->addAction("Open", this, SLOT(loadWorkflow()), QKeySequence(Qt::CTRL + Qt::Key_O));
  fileMenu->insertSeparator(fileMenu->actions().last());
  fileMenu->addAction("Save", this, SLOT(save()), QKeySequence(Qt::CTRL + Qt::Key_S));
  fileMenu->addAction("Save as", this, SLOT(saveWorkflow()));
  fileMenu->addAction("Reload", this, SLOT(reload()), QKeySequence(Qt::CTRL + Qt::Key_R));

  fileMenu->addAction("Quit", this, SLOT(quit()));
  fileMenu->insertSeparator(fileMenu->actions().last());

  editMenu = menuBar()->addMenu("&Edit");
  changeInterfaceAction = editMenu->addAction("Change Workflow Interface", this, SLOT(editCurrentInterface()), QKeySequence(Qt::Key_F4));
  changeInterfaceAction->setEnabled(true);
  connect(editMenu, SIGNAL(aboutToShow()), this, SLOT(updateEditMenuStatus()));
  connect(editMenu, SIGNAL(aboutToHide()), this, SLOT(enableEditMenuItems()));

  runMenu = menuBar()->addMenu("&Run");
  runMenu->addAction("Update", this, SLOT(updateCurrentModule()), QKeySequence(Qt::Key_F5));
  runMenu->addAction("Update All", this, SLOT(updateWorkflow()), QKeySequence(Qt::Key_F9));
  runMenu->addAction("Abort", this, SLOT(terminateUpdate()), QKeySequence(Qt::Key_Escape));

  connect(&reloadTimer, SIGNAL(timeout()), this, SLOT(checkLibraryUpdates()));
  connect(toolBox, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(itemClickedHandler(QTreeWidgetItem*, int)));
  connect(toolBox, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClickedHandler(QTreeWidgetItem*, int)));

  if (DataModel::getInstance().getAutoReload()) {
    reloadTimer.setInterval(1000);
    reloadTimer.start();
  }
}

MainWindow::~MainWindow()
{
  delete fileMenu;
  delete runMenu;
  delete editMenu;
}

void MainWindow::closeEvent(QCloseEvent *event) {
  DataModel& model = DataModel::getInstance();
  model.setWindowX(x());
  model.setWindowY(y());
  model.setWindowWidth(width());
  model.setWindowHeight(height());

  QMainWindow::closeEvent(event);
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::itemClickedHandler(QTreeWidgetItem *item, int) {
  if (item->childCount()) {
    item->setExpanded(!item->isExpanded());
  }
}

void MainWindow::itemDoubleClickedHandler(QTreeWidgetItem *item, int) {
  if (!item->childCount()) {
    // new tool
    string classname = item->data(0, Qt::UserRole).toString().toAscii().data();
    //openWorkflows[tabWidget->currentIndex()]->newModule(classname);
  }
}

void MainWindow::loadWorkflow() {
  QFileDialog fileDialog(this);

  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      openWorkflows[tabWidget->currentIndex()]->load(filenames[0].toUtf8().data());
    }
  }
}

void MainWindow::saveWorkflow() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      //Controller::getInstance().saveCurrentWorkflow(filenames[0].toUtf8().data());
      Xmlizer::ToXml(filenames[0].toUtf8().data(), *openWorkflows[tabWidget->currentIndex()]);
    }
  }
}

void MainWindow::loadLibrary() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      //Workflow* workflow = DataModel::getInstance().getMainWorkflow();
      Workflow* workflow = openWorkflows[tabWidget->currentIndex()];
      vector<string>* libs = workflow->getLibraries();
      libs->push_back(filenames[0].toUtf8().data());
      workflow->setLibraries(libs);
    }
    updateToolBox(toolBox);
  }
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  Workflow* workflow = model.getMainWorkflow();

  //TiXmlElement* workflowElement = workflow->getXml(false);
  //Xmlizer::ToXml(*workflowElement, *workflow);
  TiXmlElement* workflowElement = Xmlizer::CreateXml(*workflow);
  delete workflow;
  workflow = 0;
  tabWidget->removeTab(0); // First tab is never automatically removed
  openWorkflows.clear();    // All tabs should now be closed. Either because they were closed due to delete or because they were removed manually

  workflow = dynamic_cast<Workflow*>(Xmlizer::CreateReflectableClass(*workflowElement));
  if (!workflow)
    throw "could not reload workflow";
  model.setMainWorkflow(workflow);
  workflow->resumeFromModel();
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
  workflow->resumeViewport(); // resume after the layout stuff is done.
  openWorkflows.push_back(workflow);
  connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
  connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(closeWorkflow(workflow::Workflow*)));
  updateToolBox(toolBox);
}

void MainWindow::checkLibraryUpdates() {
  if (LibraryLoader::getInstance().librariesUpdated()) {
    cout << "Update scheduled." << endl;
    libsChanged = true;
    return;
  }
  if (libsChanged) {
    cout << "Updating libraries..." << endl;
    reload();
  }
  libsChanged = false;
}

void MainWindow::setGuiEnabled(bool enabled) {
  fileMenu->setEnabled(enabled);
  editMenu->setEnabled(enabled);
  toolBox->setEnabled(enabled);
  //for (int i = 0; i < tabWidget->count(); ++i)
  //  tabWidget->widget(i)->setEnabled(enabled);
  for (unsigned i = 0; i < openWorkflows.size(); ++i)
    openWorkflows[i]->setUiEnabled(enabled);
}

void MainWindow::updateCurrentModule() {
  setGuiEnabled(false);

  Workflow* workingWorkflow = openWorkflows[tabWidget->currentIndex()];
  connect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
  workingWorkflow->updateCurrentModule();
}

void MainWindow::updateWorkflow() {
  setGuiEnabled(false);

  Workflow* workingWorkflow = openWorkflows[tabWidget->currentIndex()];
  connect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
  workingWorkflow->updateOutputs();
}

void MainWindow::terminateUpdate() {
  setGuiEnabled(true);
}

void MainWindow::updateFinished(Node* node) {
  setGuiEnabled(true);
  Workflow* workingWorkflow = dynamic_cast<Workflow*>(node);
  if (workingWorkflow)
    disconnect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
}

void MainWindow::save() {
  DataModel::getInstance().save();
}

void MainWindow::showWorkflow(workflow::Workflow* workflow) {
  assert((int)openWorkflows.size() == tabWidget->count());

  unsigned pos = 0;
  for(;pos < openWorkflows.size(); ++pos) {
    if (openWorkflows[pos] == workflow)
      break;
  }

  if (pos < openWorkflows.size()) {
    tabWidget->setCurrentIndex(pos);
  } else {
    int currentIndex = tabWidget->count();

    tabWidget->addTab(workflow->dispenseWidget(), workflow->getToolItem()->getLabel().c_str());
    workflow->resumeViewport();
    openWorkflows.push_back(workflow);
    tabWidget->setCurrentIndex(currentIndex);
  }
}

void MainWindow::closeWorkflow(workflow::Workflow* workflow) {
  assert((int)openWorkflows.size() == tabWidget->count());

  for(unsigned pos = 0; pos < openWorkflows.size(); ++pos) {
    if (openWorkflows[pos] == workflow) {
      closeWorkflow(pos);
      break;
    }
  }
}

void MainWindow::closeWorkflow(int tabIndex) {
  if (tabIndex == 0)
    return;
  tabWidget->removeTab(tabIndex);
  openWorkflows.erase(openWorkflows.begin() + tabIndex);
}

void MainWindow::updateEditMenuStatus() {
  changeInterfaceAction->setEnabled(openWorkflows[tabWidget->currentIndex()]->getCurrentWorkflow());
}

void MainWindow::enableEditMenuItems() {
  changeInterfaceAction->setEnabled(true);
}

void MainWindow::editCurrentInterface() {
  Workflow* currentWorkflow = openWorkflows[tabWidget->currentIndex()]->getCurrentWorkflow();
  if (!currentWorkflow)
    return;

  bool newInterface = false;

  // If no current interface -> create interface, compile interface, load library, load interface class
  if(!currentWorkflow->getInterface()) {
    boost::shared_ptr<InterfaceDescription> interface(new InterfaceDescription());
    interface->setName(currentWorkflow->getInterfaceName());
    currentWorkflow->setInterface(interface);
    newInterface = true;
  }
  EditInterfaceDialog dialog(currentWorkflow->getInterface().get(), this);
  dialog.exec();
  currentWorkflow->updateInterfaceTimeStamp();
  if (newInterface)
    reload();
}

}

}
