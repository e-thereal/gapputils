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
#include <qstatusbar.h>
#include <qlabel.h>
#include <qdockwidget.h>
#include <qsettings.h>

#include "DataModel.h"
#include "Controller.h"
#include "ToolItem.h"
//#include "EditInterfaceDialog.h"
#include <capputils/ReflectableClassFactory.h>
#include <qbrush.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>

#include <iostream>
#include <boost/typeof/std/utility.hpp>

#include "WorkflowToolBox.h"
#include "PropertyGrid.h"
#include "LogbookWidget.h"
#include "GlobalPropertiesView.h"
#include "WorkbenchWindow.h"

#include <qmdiarea.h>
#include <qtextedit.h>

using namespace std;
using namespace capputils;

namespace gapputils {

using namespace workflow;

namespace host {

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags), libsChanged(false), autoQuit(false)
{
  DataModel& model = DataModel::getInstance();

  model.setMainWindow(this);

  setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
  setWindowIcon(QIcon(":/icons/application.png"));

  tabWidget = new QTabWidget();
  tabWidget->setTabsClosable(true);
  connect(tabWidget, SIGNAL(tabCloseRequested(int)), this, SLOT(closeWorkflow(int)));
  connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(currentTabChanged(int)));

  area = new QMdiArea();
  area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  area->setViewMode(QMdiArea::TabbedView);
  area->setTabsClosable(true);
  area->setTabsMovable(true);
  setCentralWidget(area);

  //area->addSubWindow(new WorkbenchWindow())->show();
//  QMdiSubWindow* sub = new QMdiSubWindow();


  fileMenu = menuBar()->addMenu("&File");
  fileMenu->addAction("Load Library", this, SLOT(loadLibrary()), QKeySequence(Qt::CTRL + Qt::Key_L));

  fileMenu->addAction("Open", this, SLOT(loadWorkflow()), QKeySequence(Qt::CTRL + Qt::Key_O));
  fileMenu->insertSeparator(fileMenu->actions().last());
  fileMenu->addAction("Save", this, SLOT(save()), QKeySequence(Qt::CTRL + Qt::Key_S));
  fileMenu->addAction("Save as", this, SLOT(saveAs()));
  fileMenu->addAction("Reload", this, SLOT(reload()), QKeySequence(Qt::CTRL + Qt::Key_R));

  fileMenu->addAction("Quit", this, SLOT(quit()));
  fileMenu->insertSeparator(fileMenu->actions().last());

  editMenu = menuBar()->addMenu("&Edit");
  editMenu->addAction("Copy", this, SLOT(copy()), QKeySequence(Qt::CTRL + Qt::Key_C));
  editMenu->addAction("Paste", this, SLOT(paste()), QKeySequence(Qt::CTRL + Qt::Key_V));

  runMenu = menuBar()->addMenu("&Run");
  runMenu->addAction("Update", this, SLOT(updateCurrentModule()), QKeySequence(Qt::Key_F5));
  runMenu->addAction("Update All", this, SLOT(updateWorkflow()), QKeySequence(Qt::Key_F9));
  abortAction = runMenu->addAction("Abort", this, SLOT(terminateUpdate()), QKeySequence(Qt::Key_Escape));
  abortAction->setEnabled(false);

  windowMenu = menuBar()->addMenu("&Window");

  connect(&reloadTimer, SIGNAL(timeout()), this, SLOT(checkLibraryUpdates()));

  if (DataModel::getInstance().getAutoReload()) {
    reloadTimer.setInterval(1000);
    reloadTimer.start();
  }

  statusBar()->showMessage("Ready.");

  statusBar()->addPermanentWidget(new QLabel("Elapsed:"), 0);
  QLabel* label = new QLabel();
  label->setFixedWidth(130);
  statusBar()->addPermanentWidget(label, 0);
  model.setPassedLabel(label);

  statusBar()->addPermanentWidget(new QLabel("Remaining:"), 0);
  label = new QLabel();
  label->setFixedWidth(130);
  statusBar()->addPermanentWidget(label, 0);
  model.setRemainingLabel(label);

  statusBar()->addPermanentWidget(new QLabel("Total:"), 0);
  label = new QLabel();
  label->setFixedWidth(130);
  statusBar()->addPermanentWidget(label, 0);
  model.setTotalLabel(label);

  statusBar()->addPermanentWidget(new QLabel("Finished by:"), 0);
  label = new QLabel();
  label->setFixedWidth(150);
  statusBar()->addPermanentWidget(label, 0);
  model.setFinishedLabel(label);

  setCorner(Qt::TopLeftCorner, Qt::LeftDockWidgetArea);
  setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
  setCorner(Qt::TopRightCorner, Qt::RightDockWidgetArea);
  setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

  QDockWidget *dock = new QDockWidget(tr("Modules"), this);
  dock->setObjectName("ModulesToolBox");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  toolBox = new WorkflowToolBox(dock);
  dock->setWidget(toolBox);
//  dock->setStyleSheet("::title { position: relative; padding-left: 50px;"
//                            "          text-align: left center }");
  addDockWidget(Qt::LeftDockWidgetArea, dock);

  windowMenu->addAction(dock->toggleViewAction());
  editMenu->addAction("Filter", toolBox, SLOT(focusFilter()), QKeySequence(Qt::CTRL + Qt::Key_F));

  dock = new QDockWidget(tr("Property Grid"), this);
  dock->setObjectName("PropertyGrid");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  propertyGrid = new PropertyGrid(dock);
  dock->setWidget(propertyGrid);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  dock = new QDockWidget(tr("Logbook"), this);
  dock->setObjectName("Logbook");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea
      | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  LogbookWidget* logbook = new LogbookWidget(dock);
  dock->setWidget(logbook);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  dock = new QDockWidget(tr("Global Properties"), this);
  dock->setObjectName("GlobalProperties");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea
      | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  GlobalPropertiesView* propView = new GlobalPropertiesView(dock);
  dock->setWidget(propView);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  connect(logbook, SIGNAL(selectModuleRequested(const QString&)),
      this, SLOT(selectModule(const QString&)));

  editMenu->addAction("Reset Inputs", this, SLOT(resetInputs()), QKeySequence(Qt::Key_Home));
  editMenu->insertSeparator(editMenu->actions().last());
  editMenu->addAction("Increment Inputs", this, SLOT(incrementInputs()), QKeySequence(Qt::Key_PageDown));
  editMenu->addAction("Decrement Inputs", this, SLOT(decrementInputs()), QKeySequence(Qt::Key_PageUp));

  //setDockNestingEnabled(true);
}

MainWindow::~MainWindow()
{
  delete fileMenu;
  delete runMenu;
  delete editMenu;
  delete windowMenu;
}

void MainWindow::closeEvent(QCloseEvent *event) {
  DataModel& model = DataModel::getInstance();
  model.setWindowX(x());
  model.setWindowY(y());
  model.setWindowWidth(width());
  model.setWindowHeight(height());
  //std::cout << "Save: " << x() << ", " << y() << ", " << width() << ", " << height() << std::endl;

  QSettings settings;
  settings.setValue("windowState", saveState());
  QMainWindow::closeEvent(event);
}

void MainWindow::resume() {
  DataModel& model = DataModel::getInstance();
  int left = model.getWindowX(), top = model.getWindowY(), w = model.getWindowWidth(), h = model.getWindowHeight();
  this->setGeometry(left, top, w, h);
  left += left - x();
  top += top - y();
  this->setGeometry(left, top, w, h);

  boost::shared_ptr<Workflow> workflow = model.getMainWorkflow();
  string currentUuid = model.getCurrentWorkflow();

  if (!currentUuid.size()) {
    currentUuid = workflow->getUuid();
    model.setCurrentWorkflow(currentUuid);
  }

  workflow->resume();
  openWorkflows.push_back(workflow);
  tabWidget->addTab(workflow->dispenseWidget(), "Main");

  area->addSubWindow(new WorkbenchWindow(workflow));

  workflow->resumeViewport(); // resume after the layout stuff is done.
  connect(workflow.get(), SIGNAL(showWorkflowRequest(boost::shared_ptr<workflow::Workflow>)), this, SLOT(showWorkflow(boost::shared_ptr<workflow::Workflow>)));
  connect(workflow.get(), SIGNAL(deleteCalled(const std::string&)), this, SLOT(closeWorkflow(const std::string&)));
  connect(workflow.get(), SIGNAL(currentModuleChanged(boost::shared_ptr<workflow::Node>)), this, SLOT(handleCurrentNodeChanged(boost::shared_ptr<workflow::Node>)));

//  std::cout << "Opening " << model.getOpenWorkflows()->size() << " workflows." << std::endl;
  for (unsigned i = 0; i < model.getOpenWorkflows()->size(); ++i) {
    string uuid = model.getOpenWorkflows()->at(i);
    assert(model.getWorkflowMap()->find(uuid) != model.getWorkflowMap()->end());
    showWorkflow(model.getWorkflowMap()->at(uuid).lock(), false);
  }
  if(model.getWorkflowMap()->find(currentUuid) != model.getWorkflowMap()->end())
    showWorkflow(model.getWorkflowMap()->at(currentUuid).lock());

  QSettings settings;
  if (settings.contains("windowState"))
    restoreState(settings.value("windowState").toByteArray());
}

void MainWindow::setAutoQuit(bool autoQuit) {
  this->autoQuit = autoQuit;
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::copy() {
  openWorkflows[tabWidget->currentIndex()].lock()->copySelectedNodesToClipboard();
}

void MainWindow::paste() {
  // TODO: use MDI
//  openWorkflows[tabWidget->currentIndex()].lock()->addNodesFromClipboard();
}

void MainWindow::loadWorkflow() {
  QString filename = QFileDialog::getOpenFileName(this, "Load configuration", "", "Host Configuration (*.xml *.config)");
  if (!filename.isNull()) {
    DataModel& model = DataModel::getInstance();
    model.setConfiguration(filename.toAscii().data());
    openWorkflows.clear();    // All tabs should now be closed. Either because they were closed due to delete or because they were removed manually
    tabWidget->removeTab(0);  // First tab is never automatically removed

    setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
    Xmlizer::FromXml(model, model.getConfiguration());
    resume();
    toolBox->update();
  }
}

void MainWindow::saveWorkflow() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      //Controller::getInstance().saveCurrentWorkflow(filenames[0].toUtf8().data());
      Xmlizer::ToXml(filenames[0].toUtf8().data(), *openWorkflows[tabWidget->currentIndex()].lock());
    }
  }
}

void MainWindow::loadLibrary() {
#ifdef _WIN32
  QString filename = QFileDialog::getOpenFileName(this, "Open Library", "", "Library (*.dll)");
#else
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "", "Library (*.so)");
#endif
  if (filename.isNull())
    return;

  boost::shared_ptr<Workflow> workflow = openWorkflows[tabWidget->currentIndex()].lock();
  boost::shared_ptr<vector<string> > libs = workflow->getLibraries();
  libs->push_back(filename.toUtf8().data());
  workflow->setLibraries(libs);
  toolBox->update();
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  TiXmlElement* modelElement = Xmlizer::CreateXml(model);
  model.setMainWorkflow(boost::shared_ptr<Workflow>());
  openWorkflows.clear();    // All tabs should now be closed. Either because they were closed due to delete or because they were removed manually
  tabWidget->removeTab(0);  // First tab is never automatically removed

  Xmlizer::FromXml(model, *modelElement);
  resume();
  toolBox->update();
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
  propertyGrid->setEnabled(enabled);
  for (unsigned i = 0; i < openWorkflows.size(); ++i)
    openWorkflows[i].lock()->setUiEnabled(enabled);
}

void MainWindow::updateCurrentModule() {
  setGuiEnabled(false);

  workingWorkflow = openWorkflows[tabWidget->currentIndex()];

  boost::shared_ptr<Workflow> workingWorkflow = this->workingWorkflow.lock();
  connect(workingWorkflow.get(), SIGNAL(updateFinished(boost::shared_ptr<workflow::Node>)), this, SLOT(updateFinished(boost::shared_ptr<workflow::Node>)));
  workingWorkflow->updateCurrentModule();
  abortAction->setEnabled(true);
}

void MainWindow::updateWorkflow() {
  setGuiEnabled(false);
  workingWorkflow = openWorkflows[tabWidget->currentIndex()];

  boost::shared_ptr<Workflow> workingWorkflow = this->workingWorkflow.lock();
  connect(workingWorkflow.get(), SIGNAL(updateFinished(boost::shared_ptr<workflow::Node>)), this, SLOT(updateFinished(boost::shared_ptr<workflow::Node>)));
  workingWorkflow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::updateMainWorkflow() {
  setGuiEnabled(false);

  workingWorkflow = openWorkflows[0];

  boost::shared_ptr<Workflow> workingWorkflow = openWorkflows[tabWidget->currentIndex()].lock();
  connect(workingWorkflow.get(), SIGNAL(updateFinished(boost::shared_ptr<workflow::Node>)), this, SLOT(updateFinished(boost::shared_ptr<workflow::Node>)));
  workingWorkflow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::terminateUpdate() {
  workingWorkflow.lock()->abortUpdate();
}

void MainWindow::updateFinished(boost::shared_ptr<Node> node) {
  abortAction->setEnabled(false);
  setGuiEnabled(true);
  boost::shared_ptr<workflow::Workflow> workflow = workingWorkflow.lock();
  assert(workflow.get() == node.get());
  if (workflow)
    disconnect(workflow.get(), SIGNAL(updateFinished(boost::shared_ptr<workflow::Node>)), this, SLOT(updateFinished(boost::shared_ptr<workflow::Node>)));

  if (autoQuit)
    quit();
}

void MainWindow::save() {
  DataModel::getInstance().save();
}

void MainWindow::saveAs() {
  QString filename = QFileDialog::getSaveFileName(this, "Save File", "", "Host Configuration (*.xml)");
  if (!filename.isNull()) {
    DataModel& model = DataModel::getInstance();
    model.setConfiguration(filename.toUtf8().data());
    model.save();
    setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
  }
}

void MainWindow::showWorkflow(boost::shared_ptr<workflow::Workflow> workflow, bool addUuid) {
  assert((int)openWorkflows.size() == tabWidget->count());

  unsigned pos = 0;
  for(;pos < openWorkflows.size(); ++pos) {
    if (openWorkflows[pos].lock() == workflow)
      break;
  }

  if (pos < openWorkflows.size()) {
    tabWidget->setCurrentIndex(pos);
  } else {
    int currentIndex = tabWidget->count();
    openWorkflows.push_back(workflow);
    tabWidget->addTab(workflow->dispenseWidget(), workflow->getToolItem()->getLabel().c_str());
    workflow->resumeViewport();
    if (addUuid)
      DataModel::getInstance().getOpenWorkflows()->push_back(workflow->getUuid());
    tabWidget->setCurrentIndex(currentIndex);
    connect(workflow.get(), SIGNAL(currentModuleChanged(boost::shared_ptr<workflow::Node>)), this, SLOT(handleCurrentNodeChanged(boost::shared_ptr<workflow::Node>)));
  }
}

void MainWindow::closeWorkflow(const std::string& uuid) {
//  std::cout << "Closing workflow" << std::endl;
  assert((int)openWorkflows.size() == tabWidget->count());

  for(unsigned pos = 0; pos < openWorkflows.size(); ++pos) {
    if (openWorkflows[pos].expired() || openWorkflows[pos].lock()->getUuid() == uuid) {
      closeWorkflow(pos);
    }
  }
}

void MainWindow::closeWorkflow(int tabIndex) {
  vector<string>* openWorkflowUuids = DataModel::getInstance().getOpenWorkflows().get();
  assert(openWorkflows.size() == openWorkflowUuids->size() + 1);
  if (tabIndex == 0)
    return;

  boost::shared_ptr<Workflow> workflow = openWorkflows[tabIndex].lock();
  if (workflow)
    disconnect(workflow.get(), SIGNAL(currentModuleChanged(boost::shared_ptr<workflow::Node>)), this, SLOT(handleCurrentNodeChanged(boost::shared_ptr<workflow::Node>)));

  tabWidget->removeTab(tabIndex);
  openWorkflows.erase(openWorkflows.begin() + tabIndex);
  openWorkflowUuids->erase(openWorkflowUuids->begin() + (tabIndex - 1));
}

void MainWindow::currentTabChanged(int index) {
  if (index >= 0 && index < (int)openWorkflows.size()) {
    boost::shared_ptr<Workflow> workflow = openWorkflows[index].lock();
    DataModel::getInstance().setCurrentWorkflow(workflow->getUuid());
    propertyGrid->setNode(workflow->getCurrentNode());
  }
}

void MainWindow::handleCurrentNodeChanged(boost::shared_ptr<workflow::Node> node) {
  propertyGrid->setNode(node);
}

void MainWindow::selectModule(const QString& quuid) {
  std::string uuid = quuid.toAscii().data();

  for (unsigned i = 0; i < openWorkflows.size(); ++i) {
    if (openWorkflows[i].lock()->trySelectNode(uuid)) {
      tabWidget->setCurrentIndex(i);
      break;
    }
  }
}

void MainWindow::resetInputs() {
  openWorkflows[tabWidget->currentIndex()].lock()->resetInputs();
}

void MainWindow::incrementInputs() {
  openWorkflows[tabWidget->currentIndex()].lock()->incrementInputs();
}

void MainWindow::decrementInputs() {
  openWorkflows[tabWidget->currentIndex()].lock()->decrementInputs();
}

}

}
