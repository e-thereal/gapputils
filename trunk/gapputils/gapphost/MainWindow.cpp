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
#include <capputils/ReflectableClassFactory.h>
#include <qbrush.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>

#include <iostream>
#include <boost/typeof/std/utility.hpp>
#include <boost/make_shared.hpp>

#include "WorkflowToolBox.h"
#include "WorkflowSnippets.h"
#include "PropertyGrid.h"
#include "LogbookWidget.h"
#include "GlobalPropertiesView.h"
#include "WorkbenchWindow.h"
#include "ModuleHelpWidget.h"

#include <qmdiarea.h>
#include <qtextedit.h>

using namespace std;
using namespace capputils;

namespace gapputils {

using namespace workflow;

namespace host {

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags), autoQuit(false), workingWindow(0)
{
  DataModel& model = DataModel::getInstance();

  model.setMainWindow(this);

#ifdef _RELEASE
  setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
#else
  setWindowTitle(QString("grapevine (debug mode) - ") + model.getConfiguration().c_str());
#endif
  setWindowIcon(QIcon(":/icons/application.png"));

  area = new QMdiArea();
  area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  area->setViewMode(QMdiArea::TabbedView);
  area->setTabsClosable(true);
  area->setTabsMovable(true);
  setCentralWidget(area);
  connect(area, SIGNAL(subWindowActivated(QMdiSubWindow*)), this, SLOT(subWindowActivated(QMdiSubWindow*)));

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
  editMenu->addAction("Copy (including dangling edges)", this, SLOT(copyDanglingEdges()), QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_C));
  editMenu->addAction("Paste", this, SLOT(paste()), QKeySequence(Qt::CTRL + Qt::Key_V));
  editMenu->addAction("Delete", this, SLOT(removeSelectedItems()));
  editMenu->addAction("Create Workflow Snipped", this, SLOT(createSnippet()), QKeySequence(Qt::CTRL + Qt::Key_N));

  runMenu = menuBar()->addMenu("&Run");
  runMenu->addAction("Update", this, SLOT(updateCurrentModule()), QKeySequence(Qt::Key_F5));
  runMenu->addAction("Update All", this, SLOT(updateWorkflow()), QKeySequence(Qt::Key_F9));
  abortAction = runMenu->addAction("Abort", this, SLOT(terminateUpdate()), QKeySequence(Qt::Key_Escape));
  abortAction->setEnabled(false);

  windowMenu = menuBar()->addMenu("&Window");

  connect(&autoSaveTimer, SIGNAL(timeout()), this, SLOT(autoSave()));

//  if (DataModel::getInstance().getAutoReload()) {
  if (model.getSaveConfiguration()) {
    autoSaveTimer.setInterval(60000);
//    autoSaveTimer.start();
  }
//  }

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

  // Modules toolbox
  QDockWidget *dock = new QDockWidget(tr("Modules"), this);
  dock->setObjectName("ModulesToolBox");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
//  toolBox = new WorkflowToolBox(dock);
  toolBox = &WorkflowToolBox::GetInstance();
  toolBox->setParent(dock);
  dock->setWidget(toolBox);
//  dock->setStyleSheet("::title { position: relative; padding-left: 50px;"
//                            "          text-align: left center }");
  addDockWidget(Qt::LeftDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());
  editMenu->addAction("Filter", toolBox, SLOT(focusFilter()), QKeySequence(Qt::CTRL + Qt::Key_F));

  // Workflow Snippets
  dock = new QDockWidget(tr("Snippets"), this);
  dock->setObjectName("WorkflowSnippets");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  snippets = &WorkflowSnippets::GetInstance();
  dock->setWidget(snippets);
//  dock->setStyleSheet("::title { position: relative; padding-left: 50px;"
//                            "          text-align: left center }");
  addDockWidget(Qt::LeftDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  // Property Grid
  dock = new QDockWidget(tr("Property Grid"), this);
  dock->setObjectName("PropertyGrid");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  propertyGrid = new PropertyGrid(dock);
  dock->setWidget(propertyGrid);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  // Logbook
  dock = new QDockWidget(tr("Logbook"), this);
  dock->setObjectName("Logbook");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea
      | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  LogbookWidget* logbook = new LogbookWidget(dock);
  dock->setWidget(logbook);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  // Global Properties
  dock = new QDockWidget(tr("Global Properties"), this);
  dock->setObjectName("GlobalProperties");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea
      | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
  globalPropertiesView = new GlobalPropertiesView(dock);
  dock->setWidget(globalPropertiesView);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  // Module Help
  dock = new QDockWidget(tr("Help"), this);
  dock->setObjectName("ModuleHelp");
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  moduleHelp = new ModuleHelpWidget(dock);
  dock->setWidget(moduleHelp);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  windowMenu->addAction(dock->toggleViewAction());

  connect(logbook, SIGNAL(selectModuleRequested(const QString&)),
      this, SLOT(selectModule(const QString&)));
  connect(globalPropertiesView, SIGNAL(selectModuleRequested(const QString&)),
      this, SLOT(selectModule(const QString&)));
  connect(propertyGrid, SIGNAL(selectModuleRequested(const QString&)),
      this, SLOT(selectModule(const QString&)));

  connect(toolBox, SIGNAL(itemSelected(QString)), moduleHelp, SLOT(setClassname(QString)));

  windowMenu->addAction("Tile Windows", area, SLOT(tileSubWindows()));
  windowMenu->insertSeparator(windowMenu->actions().last());

  editMenu->addAction("Reset Inputs", this, SLOT(resetInputs()), QKeySequence(Qt::Key_Home));
  editMenu->insertSeparator(editMenu->actions().last());
  editMenu->addAction("Increment Inputs", this, SLOT(incrementInputs()), QKeySequence(Qt::Key_PageDown));
  editMenu->addAction("Decrement Inputs", this, SLOT(decrementInputs()), QKeySequence(Qt::Key_PageUp));

  //setDockNestingEnabled(true);
}

MainWindow::~MainWindow() {
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

  saveWorkflowList();

  //std::cout << "Save: " << x() << ", " << y() << ", " << width() << ", " << height() << std::endl;

  QSettings settings;
  settings.setValue("windowState", saveState());
  QMainWindow::closeEvent(event);

  autoSaveTimer.stop();
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

  grandpa = boost::make_shared<workflow::Workflow>();
  grandpa->getNodes()->push_back(workflow);
  grandpa->resume();
//  workflow->resume();
  showWorkflow(workflow)->setClosable(false);

  globalPropertiesView->setWorkflow(workflow);

  for (unsigned i = 0; i < model.getOpenWorkflows()->size(); ++i) {
    string uuid = model.getOpenWorkflows()->at(i);
    if (model.getWorkflowMap()->find(uuid) != model.getWorkflowMap()->end())
      showWorkflow(model.getWorkflowMap()->at(uuid).lock());
  }

  if(model.getWorkflowMap()->find(currentUuid) != model.getWorkflowMap()->end())
    showWorkflow(model.getWorkflowMap()->at(currentUuid).lock());

  QSettings settings;
  if (settings.contains("windowState"))
    restoreState(settings.value("windowState").toByteArray());
}

WorkbenchWindow* MainWindow::showWorkflow(boost::shared_ptr<workflow::Workflow> workflow) {
  Q_FOREACH (QMdiSubWindow *window, area->subWindowList()) {
    WorkbenchWindow* wwindow = dynamic_cast<WorkbenchWindow*>(window);
    if (wwindow && wwindow->getWorkflow() == workflow) {
      area->setActiveSubWindow(window);
      return wwindow;
    }
  }

  WorkbenchWindow* window = new WorkbenchWindow(workflow);
  area->addSubWindow(window);
  window->show();
  window->resumeViewport();
  window->setUiEnabled(!workingWindow);

  return window;
}

void MainWindow::saveWorkflowList() {
  DataModel& model = DataModel::getInstance();

  boost::shared_ptr<std::vector<std::string> > workflows(new std::vector<std::string>());
  Q_FOREACH (QMdiSubWindow *w, area->subWindowList()) {
    WorkbenchWindow* window = dynamic_cast<WorkbenchWindow*>(w);
    if (window && window->getWorkflow()) {
      workflows->push_back(window->getWorkflow()->getUuid());
    }
  }
  model.setOpenWorkflows(workflows);
  model.setCurrentWorkflow(static_cast<WorkbenchWindow*>(area->currentSubWindow())->getWorkflow()->getUuid());
}

WorkbenchWindow* MainWindow::getCurrentWorkbenchWindow() {
  return static_cast<WorkbenchWindow*>(area->currentSubWindow());
}

void MainWindow::setAutoQuit(bool autoQuit) {
  this->autoQuit = autoQuit;
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::copy() {
  getCurrentWorkbenchWindow()->copySelectedNodesToClipboard();
}

void MainWindow::copyDanglingEdges() {
  getCurrentWorkbenchWindow()->copySelectedNodesToClipboard(true);
}

void MainWindow::paste() {
  getCurrentWorkbenchWindow()->addNodesFromClipboard();
}

void MainWindow::removeSelectedItems() {
  getCurrentWorkbenchWindow()->removeSelectedItems();
}

void MainWindow::createSnippet() {
  getCurrentWorkbenchWindow()->createSnippet();
}

void MainWindow::loadWorkflow() {
  QString filename = QFileDialog::getOpenFileName(this, "Load configuration", "", "Host Configuration (*.xml *.config)");
  if (!filename.isNull()) {
    DataModel& model = DataModel::getInstance();

    model.setMainWorkflow(boost::shared_ptr<workflow::Workflow>());
    model.getOpenWorkflows()->clear();
    model.setConfiguration(filename.toAscii().data());

#ifdef _RELEASE
    setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
#else
    setWindowTitle(QString("grapevine (debug mode) - ") + model.getConfiguration().c_str());
#endif
    Xmlizer::FromXml(model, model.getConfiguration());
    resume();
    toolBox->update();
  }
}

void MainWindow::saveWorkflow() {
  saveWorkflowList();
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      Xmlizer::ToXml(filenames[0].toUtf8().data(), *static_cast<WorkbenchWindow*>(area->currentSubWindow())->getWorkflow());
    }
  }
}

void MainWindow::loadLibrary() {
  DataModel& model = DataModel::getInstance();
#ifdef _WIN32
  QString filename = QFileDialog::getOpenFileName(this, "Open Library", "", "Library (*.dll)");
#else
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "", "Library (*.so)");
#endif
  if (filename.isNull())
    return;

  WorkbenchWindow* window = dynamic_cast<WorkbenchWindow*>(area->currentSubWindow());
  if (!window) {
    std::cout << area->currentSubWindow() << std::endl;
    std::cout << window << std::endl;
    return;
  }
  boost::shared_ptr<Workflow> workflow = window->getWorkflow();
  boost::shared_ptr<vector<string> > libs = workflow->getLibraries();

  std::string path = model.getLibraryPath();
  if (path.size()) {
    if (filename.startsWith(path.c_str())) {
      filename = filename.right(filename.size() - path.size());
      if (filename[0] == '/')
        filename = filename.right(filename.size() - 1);
    }
  }

  libs->push_back(filename.toUtf8().data());
  workflow->setLibraries(libs);
  toolBox->update();
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  TiXmlElement* modelElement = Xmlizer::CreateXml(model);

  saveWorkflowList();

  // close all windows
  model.setMainWorkflow(boost::shared_ptr<Workflow>());

  Xmlizer::FromXml(model, *modelElement);
  resume();
  toolBox->update();
}

//void MainWindow::checkLibraryUpdates() {
//  if (LibraryLoader::getInstance().librariesUpdated()) {
//    cout << "Update scheduled." << endl;
//    libsChanged = true;
//    return;
//  }
//  if (libsChanged) {
//    cout << "Updating libraries..." << endl;
//    reload();
//  }
//  libsChanged = false;
//}

void MainWindow::autoSave() {
  statusBar()->showMessage("Saving configuration ...");
  DataModel::getInstance().save(DataModel::AutoSaveName);
  statusBar()->showMessage("Ready.");
}

void MainWindow::setGuiEnabled(bool enabled) {
  fileMenu->setEnabled(enabled);
  editMenu->setEnabled(enabled);
  toolBox->setEnabled(enabled);
  snippets->setEnabled(enabled);
  propertyGrid->setEnabled(enabled);

  Q_FOREACH (QMdiSubWindow *w, area->subWindowList()) {
    WorkbenchWindow* window = static_cast<WorkbenchWindow*>(w);
    window->setUiEnabled(enabled);
  }
}

void MainWindow::updateCurrentModule() {
  setGuiEnabled(false);

  workingWindow = static_cast<WorkbenchWindow*>(area->currentSubWindow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateCurrentModule();
  abortAction->setEnabled(true);
}

void MainWindow::updateWorkflow() {
  setGuiEnabled(false);
  
  workingWindow = static_cast<WorkbenchWindow*>(area->currentSubWindow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::updateMainWorkflow() {
  setGuiEnabled(false);

  workingWindow = showWorkflow(DataModel::getInstance().getMainWorkflow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::updateMainWorkflowNode(const std::string& nodeLabel) {
  setGuiEnabled(false);

  workingWindow = showWorkflow(DataModel::getInstance().getMainWorkflow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateNodeByLabel(nodeLabel);
  abortAction->setEnabled(true);
}

void MainWindow::updateMainWorkflowNodes(const std::vector<std::string>& nodeLabels) {
  setGuiEnabled(false);

  workingWindow = showWorkflow(DataModel::getInstance().getMainWorkflow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateNodesByLabels(nodeLabels);
  abortAction->setEnabled(true);
}

void MainWindow::updateCurrentWorkflowNode(const capputils::reflection::ReflectableClass* object) {
  setGuiEnabled(false);

  workingWindow = static_cast<WorkbenchWindow*>(area->currentSubWindow());
  connect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
  workingWindow->updateNode(object);
  abortAction->setEnabled(true);
}

void MainWindow::terminateUpdate() {
  workingWindow->abortUpdate();
}

void MainWindow::updateFinished() {
  abortAction->setEnabled(false);
  setGuiEnabled(true);
  if (workingWindow)
    disconnect(workingWindow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));

  if (autoQuit)
    quit();
  workingWindow = 0;
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
#ifdef _RELEASE
    setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
#else
    setWindowTitle(QString("grapevine (debug mode) - ") + model.getConfiguration().c_str());
#endif
  }
}

void MainWindow::closeWorkflow(const std::string& uuid) {
//  std::cout << "Closing workflow" << std::endl;
  Q_FOREACH (QMdiSubWindow *w, area->subWindowList()) {
    WorkbenchWindow* window = static_cast<WorkbenchWindow*>(w);
    if (window->getWorkflow()->getUuid() == uuid) {
      area->removeSubWindow(window);
      delete window;
    }
  }
}

void MainWindow::subWindowActivated(QMdiSubWindow* w) {
  if (!w)
    return;
  WorkbenchWindow* window = static_cast<WorkbenchWindow*>(w);
  boost::shared_ptr<Workflow> workflow = window->getWorkflow();
  if (!workflow)
    return;
  propertyGrid->setNode(window->getCurrentNode());
  moduleHelp->setNode(window->getCurrentNode());
  globalPropertiesView->setWorkflow(workflow);
}

void MainWindow::handleCurrentNodeChanged(boost::shared_ptr<workflow::Node> node) {
  propertyGrid->setNode(node);
  moduleHelp->setNode(node);
}

void MainWindow::selectModule(const QString& quuid) {
  std::string uuid = quuid.toAscii().data();

  Q_FOREACH (QMdiSubWindow *w, area->subWindowList()) {
    WorkbenchWindow* window = static_cast<WorkbenchWindow*>(w);
    if (window->trySelectNode(uuid)) {
      area->setActiveSubWindow(w);
      break;
    }
  }
}

void MainWindow::resetInputs() {
  static_cast<WorkbenchWindow*>(area->currentSubWindow())->updateInputs();
  static_cast<WorkbenchWindow*>(area->currentSubWindow())->getWorkflow()->resetInputs();
}

void MainWindow::incrementInputs() {
  static_cast<WorkbenchWindow*>(area->currentSubWindow())->getWorkflow()->incrementInputs();
}

void MainWindow::decrementInputs() {
  static_cast<WorkbenchWindow*>(area->currentSubWindow())->getWorkflow()->decrementInputs();
}

}

}
