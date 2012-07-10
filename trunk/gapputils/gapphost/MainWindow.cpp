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

#include "DataModel.h"
#include "Controller.h"
#include "ToolItem.h"
#include "EditInterfaceDialog.h"
#include <capputils/ReflectableClassFactory.h>
#include <qbrush.h>
#include <gapputils/WorkflowElement.h>
#include <gapputils/WorkflowInterface.h>

#include <iostream>
#include <boost/typeof/std/utility.hpp>

using namespace std;
using namespace capputils;

#define ONLY_WORKFLOWELEMENTS

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

  //QFont font = item->font(0);
  //font.setBold(true);
  //item->setFont(0, font);

  return item;
}

QTreeWidgetItem* newTool(const string& name, const string& classname) {
  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, name.c_str());
  item->setData(0, Qt::UserRole, QVariant::fromValue(QString(classname.c_str())));
  
  return item;
}

void updateToolBox(QTreeWidget* toolBox, std::map<QTreeWidgetItem*, boost::shared_ptr<std::vector<QTreeWidgetItem*> > >& toolBoxItems) {
  toolBoxItems.clear();

  toolBox->setIndentation(10);
  toolBox->setHeaderHidden(true);
#if TREEVIEW
  toolBox->setRootIsDecorated(true);
#else
  toolBox->setRootIsDecorated(false);
#endif
  toolBox->clear();

  reflection::ReflectableClassFactory& factory = reflection::ReflectableClassFactory::getInstance();
  vector<string> classNames = factory.getClassNames();
  sort(classNames.begin(), classNames.end());

  QTreeWidgetItem* item = 0;
  string groupString("");
  //toolBox->addTopLevelItem(item);

  for (unsigned i = 0; i < classNames.size(); ++i) {
    string name = classNames[i];
    string currentGroupString;

#ifdef ONLY_WORKFLOWELEMENTS
    reflection::ReflectableClass* object = factory.newInstance(name);
    if (dynamic_cast<WorkflowElement*>(object) == 0 && dynamic_cast<WorkflowInterface*>(object) == 0) {
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

    if (!currentGroupString.compare("gapputils::host::internal") ||
        !currentGroupString.compare("gapputils::workflow"))
    {
      continue;
    }

#ifdef TREEVIEW

    // search for the longest match
    cout << "GroupString: " << groupString << std::endl;
    cout << "CurrentString: " << currentGroupString << std::endl;

    int maxPos1 = currentGroupString.find("::");
    int maxPos2 = groupString.find("::");
    int maxPos = 0;

    if (maxPos1 == string::npos)
      maxPos1 = currentGroupString.size();

    if (maxPos2 == string::npos)
      maxPos2 = groupString.size();

    cout << "Comparing: '" << currentGroupString.substr(0, maxPos1) << "' to '" << groupString.substr(0, maxPos2) << "'" << std::endl;

    while (currentGroupString.substr(0, maxPos1) == groupString.substr(0, maxPos2))
    {
      maxPos = maxPos1;
      if (maxPos1 == currentGroupString.size() ||
          maxPos2 == groupString.size())
      {
        break;
      }

      maxPos1 = currentGroupString.find("::", maxPos1 + 1);
      maxPos2 = groupString.find("::", maxPos2 + 1);

      if (maxPos1 == string::npos)
        maxPos1 = currentGroupString.size();

      if (maxPos2 == string::npos)
        maxPos2 = groupString.size();

      cout << "Comparing: '" << currentGroupString.substr(0, maxPos1) << "' to '" << groupString.substr(0, maxPos2) << "'" << std::endl;
    }

    cout << "LongestMatch: " << currentGroupString.substr(0, maxPos) << std::endl;

    // Wie viele colons sind noch im current string drin. Ergo wie oft muss ich zurueck?
    maxPos2 = maxPos;
    int backSteps = (groupString.size() == maxPos2 ? 0 : 1);
    while ((maxPos2 = groupString.find("::", maxPos2 + 1)) != string::npos)
      ++backSteps;
    cout << "BackSteps: " << backSteps << std::endl;

    for (int i = 0; i < backSteps && item; ++i)
      item = item->parent();

    // 
    maxPos1 = maxPos;
    if (maxPos1 < currentGroupString.size()) {
      while ((maxPos1 = currentGroupString.find("::", maxPos1 + 1)) != string::npos) {
        cout << "New dir: " << currentGroupString.substr(0, maxPos1) << std::endl;
        QTreeWidgetItem* newItem = newCategory(currentGroupString.substr(0, maxPos1));
        if (item == 0)
          toolBox->addTopLevelItem(newItem);
        else
          item->addChild(newItem);
        item = newItem;
      }
      cout << "New dir: " << currentGroupString << std::endl;
      QTreeWidgetItem* newItem = newCategory(currentGroupString);
      if (item == 0)
        toolBox->addTopLevelItem(newItem);
      else
        item->addChild(newItem);
      item = newItem;
    }

    cout << std::endl;
    groupString = currentGroupString;

#else
    if (currentGroupString.compare(groupString)) {
      groupString = currentGroupString;
      item = newCategory(groupString);
      toolBox->addTopLevelItem(item);
      toolBoxItems[item] = boost::shared_ptr<std::vector<QTreeWidgetItem*> >(new std::vector<QTreeWidgetItem*>());
    }
#endif

    if (item) {
      QTreeWidgetItem* toolItem = newTool(name.substr(pos+1), name);
      toolBoxItems[item]->push_back(toolItem);
      item->addChild(toolItem);
    } else {
      toolBox->addTopLevelItem(newTool(name.substr(pos+1), name));
    }
  }
  //toolBox->expandAll();
}

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags), libsChanged(false), autoQuit(false), workingWorkflow(0)
{
  DataModel& model = DataModel::getInstance();

  model.setMainWindow(this);

  setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
  setWindowIcon(QIcon(":/icons/icon.png"));

  newObjectDialog = new NewObjectDialog();

  tabWidget = new QTabWidget();
  tabWidget->setTabsClosable(true);
  connect(tabWidget, SIGNAL(tabCloseRequested(int)), this, SLOT(closeWorkflow(int)));
  connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(currentTabChanged(int)));

  toolBoxFilterEdit = new QLineEdit();

  toolBox = new QTreeWidget();
  toolBox->setDragEnabled(true);
  updateToolBox(toolBox, toolBoxItems);
  filterToolBox("");

  QVBoxLayout* toolBoxLayout = new QVBoxLayout();
  //toolBoxLayout->setMargin(5);
  toolBoxLayout->addWidget(toolBoxFilterEdit);
  toolBoxLayout->addWidget(toolBox);

  QWidget* toolBoxWidget = new QWidget();
  toolBoxWidget->setLayout(toolBoxLayout);
  
  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(toolBoxWidget);
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
  fileMenu->addAction("Save as", this, SLOT(saveAs()));
  fileMenu->addAction("Reload", this, SLOT(reload()), QKeySequence(Qt::CTRL + Qt::Key_R));

  fileMenu->addAction("Quit", this, SLOT(quit()));
  fileMenu->insertSeparator(fileMenu->actions().last());

  editMenu = menuBar()->addMenu("&Edit");
  editMenu->addAction("Copy", this, SLOT(copy()), QKeySequence(Qt::CTRL + Qt::Key_C));
  editMenu->insertSeparator(editMenu->actions().last());
  editMenu->addAction("Paste", this, SLOT(paste()), QKeySequence(Qt::CTRL + Qt::Key_V));
  editMenu->addAction("Filter", this, SLOT(focusFilter()), QKeySequence(Qt::CTRL + Qt::Key_F));

  runMenu = menuBar()->addMenu("&Run");
  runMenu->addAction("Update", this, SLOT(updateCurrentModule()), QKeySequence(Qt::Key_F5));
  runMenu->addAction("Update All", this, SLOT(updateWorkflow()), QKeySequence(Qt::Key_F9));
  abortAction = runMenu->addAction("Abort", this, SLOT(terminateUpdate()), QKeySequence(Qt::Key_Escape));
  abortAction->setEnabled(false);

  connect(&reloadTimer, SIGNAL(timeout()), this, SLOT(checkLibraryUpdates()));
  connect(toolBox, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(itemClickedHandler(QTreeWidgetItem*, int)));
  connect(toolBoxFilterEdit, SIGNAL(textChanged(const QString&)), this, SLOT(filterToolBox(const QString&)));

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
}

MainWindow::~MainWindow()
{
  delete fileMenu;
  delete runMenu;
  delete editMenu;
}

void MainWindow::focusFilter() {
  toolBoxFilterEdit->setFocus();
  toolBoxFilterEdit->selectAll();
}

void MainWindow::filterToolBox(const QString& text) {
  // Do the filtering stuff here

  if (text.length()) {
    for (BOOST_AUTO(item, toolBoxItems.begin()); item != toolBoxItems.end(); ++item) {
      item->first->takeChildren();
      int count = 0;
      for (BOOST_AUTO(child, item->second->begin()); child != item->second->end(); ++child) {
        if ((*child)->text(0).contains(text)) {
          ++count;
          item->first->addChild(*child);
        }
      }
      QString label = item->first->text(0);
      if (label.contains(' '))
        label.remove(label.indexOf(' '), label.length());
      item->first->setText(0, label + " (" + QString::number(count) + ")");
    }
  } else {
    for (BOOST_AUTO(item, toolBoxItems.begin()); item != toolBoxItems.end(); ++item) {
      item->first->takeChildren();
      int count = 0;
      for (BOOST_AUTO(child, item->second->begin()); child != item->second->end(); ++child) {
        item->first->addChild(*child);
        ++count;
      }
      QString label = item->first->text(0);
      if (label.contains(' '))
        label.remove(label.indexOf(' '), label.length());
      item->first->setText(0, label + " (" + QString::number(count) + ")");
    }
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  DataModel& model = DataModel::getInstance();
  model.setWindowX(x());
  model.setWindowY(y());
  model.setWindowWidth(width());
  model.setWindowHeight(height());
  //std::cout << "Save: " << x() << ", " << y() << ", " << width() << ", " << height() << std::endl;

  QMainWindow::closeEvent(event);
}

void MainWindow::resume() {
  DataModel& model = DataModel::getInstance();
  int left = model.getWindowX(), top = model.getWindowY(), w = model.getWindowWidth(), h = model.getWindowHeight();
  this->setGeometry(left, top, w, h);
  //std::cout << "Target: " << left << ", " << top << ", " << w << ", " << h << std::endl;
  //std::cout << "Current: " << x() << ", " << y() << ", " << width() << ", " << height() << std::endl;
  left += left - x();
  top += top - y();
  this->setGeometry(left, top, w, h);
  //std::cout << "Corrected: " << x() << ", " << y() << ", " << width() << ", " << height() << std::endl;
  Workflow* workflow = model.getMainWorkflow();
  string currentUuid = model.getCurrentWorkflow();

  if (!currentUuid.size()) {
    currentUuid = workflow->getUuid();
    model.setCurrentWorkflow(currentUuid);
  }

  workflow->resume();
  openWorkflows.push_back(workflow);
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
  workflow->resumeViewport(); // resume after the layout stuff is done.
  connect(workflow, SIGNAL(showWorkflowRequest(workflow::Workflow*)), this, SLOT(showWorkflow(workflow::Workflow*)));
  connect(workflow, SIGNAL(deleteCalled(workflow::Workflow*)), this, SLOT(closeWorkflow(workflow::Workflow*)));

  for (unsigned i = 0; i < model.getOpenWorkflows()->size(); ++i) {
    string uuid = model.getOpenWorkflows()->at(i);
    assert(model.getWorkflowMap()->find(uuid) != model.getWorkflowMap()->end());
    showWorkflow(model.getWorkflowMap()->at(uuid), false);
  }
  if(model.getWorkflowMap()->find(currentUuid) != model.getWorkflowMap()->end())
    showWorkflow(model.getWorkflowMap()->at(currentUuid));
}

void MainWindow::setAutoQuit(bool autoQuit) {
  this->autoQuit = autoQuit;
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::itemClickedHandler(QTreeWidgetItem *item, int) {
  if (item->childCount()) {
    item->setExpanded(!item->isExpanded());
  }
}

void MainWindow::copy() {
  openWorkflows[tabWidget->currentIndex()]->copySelectedNodesToClipboard();
}

void MainWindow::paste() {
  openWorkflows[tabWidget->currentIndex()]->addNodesFromClipboard();
}

void MainWindow::loadWorkflow() {
  QString filename = QFileDialog::getOpenFileName(this, "Load configuration", "", "Host Configuration (*.xml *.config)");
  if (!filename.isNull()) {
    DataModel& model = DataModel::getInstance();
    model.setConfiguration(filename.toAscii().data());
    Workflow* workflow = model.getMainWorkflow();
    delete workflow;
    workflow = 0;
    openWorkflows.clear();    // All tabs should now be closed. Either because they were closed due to delete or because they were removed manually
    tabWidget->removeTab(0);  // First tab is never automatically removed

    setWindowTitle(QString("grapevine - ") + model.getConfiguration().c_str());
    Xmlizer::FromXml(model, model.getConfiguration());
    resume();
    updateToolBox(toolBox, toolBoxItems);
    filterToolBox("");
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
#ifdef _WIN32
  QString filename = QFileDialog::getOpenFileName(this, "Open Library", "", "Library (*.dll)");
#else
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "", "Library (*.so)");
#endif
  if (filename.isNull())
    return;

  Workflow* workflow = openWorkflows[tabWidget->currentIndex()];
  vector<string>* libs = workflow->getLibraries();
  libs->push_back(filename.toUtf8().data());
  workflow->setLibraries(libs);
  updateToolBox(toolBox, toolBoxItems);
  filterToolBox("");
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  Workflow* workflow = model.getMainWorkflow();

  TiXmlElement* modelElement = Xmlizer::CreateXml(model);
  delete workflow;
  workflow = 0;
  openWorkflows.clear();    // All tabs should now be closed. Either because they were closed due to delete or because they were removed manually
  tabWidget->removeTab(0);  // First tab is never automatically removed

  Xmlizer::FromXml(model, *modelElement);
  resume();
  updateToolBox(toolBox, toolBoxItems);
  filterToolBox("");
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
  for (unsigned i = 0; i < openWorkflows.size(); ++i)
    openWorkflows[i]->setUiEnabled(enabled);
}

void MainWindow::updateCurrentModule() {
  setGuiEnabled(false);

  workingWorkflow = openWorkflows[tabWidget->currentIndex()];
  connect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
  workingWorkflow->updateCurrentModule();
  abortAction->setEnabled(true);
}

void MainWindow::updateWorkflow() {
  setGuiEnabled(false);

  workingWorkflow = openWorkflows[tabWidget->currentIndex()];
  connect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
  workingWorkflow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::updateMainWorkflow() {
  setGuiEnabled(false);

  workingWorkflow = openWorkflows[0];
  connect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));
  workingWorkflow->updateOutputs();
  abortAction->setEnabled(true);
}

void MainWindow::terminateUpdate() {
  workingWorkflow->abortUpdate();
}

void MainWindow::updateFinished(Node* node) {
  abortAction->setEnabled(false);
  setGuiEnabled(true);
  assert(workingWorkflow == dynamic_cast<Workflow*>(node));
  if (workingWorkflow)
    disconnect(workingWorkflow, SIGNAL(updateFinished(workflow::Node*)), this, SLOT(updateFinished(workflow::Node*)));

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

void MainWindow::showWorkflow(workflow::Workflow* workflow, bool addUuid) {
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
    openWorkflows.push_back(workflow);
    tabWidget->addTab(workflow->dispenseWidget(), workflow->getToolItem()->getLabel().c_str());
    workflow->resumeViewport();
    if (addUuid)
      DataModel::getInstance().getOpenWorkflows()->push_back(workflow->getUuid());
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
  vector<string>* openWorkflowUuids = DataModel::getInstance().getOpenWorkflows().get();
  assert(openWorkflows.size() == openWorkflowUuids->size() + 1);
  if (tabIndex == 0)
    return;
  tabWidget->removeTab(tabIndex);
  openWorkflows.erase(openWorkflows.begin() + tabIndex);
  openWorkflowUuids->erase(openWorkflowUuids->begin() + (tabIndex - 1));
}

void MainWindow::currentTabChanged(int index) {
  if (index >= 0 && index < (int)openWorkflows.size())
    DataModel::getInstance().setCurrentWorkflow(openWorkflows[index]->getUuid());
}

}

}
