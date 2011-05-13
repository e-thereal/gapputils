#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <iostream>
#include <qsplitter.h>

#include <Xmlizer.h>
#include <LibraryLoader.h>
#include <qtoolbox.h>
#include <qtreewidget.h>

#include "DataModel.h"
#include "Controller.h"
#include <ReflectableClassFactory.h>
#include <qbrush.h>
#include <WorkflowElement.h>

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

QTreeWidgetItem* newTool(const string& name) {
  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, name.c_str());
  
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
    if (pos != string::npos) {
      currentGroupString = name.substr(0, pos-1);
    } else {
      pos = -1;
    }

    if (currentGroupString.compare(groupString)) {
      groupString = currentGroupString;
      item = newCategory(groupString);
      toolBox->addTopLevelItem(item);
    }

    item->addChild(newTool(name.substr(pos+1)));
  }
  

  toolBox->expandAll();
}

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags), libsChanged(false)
{
  Workflow* workflow = DataModel::getInstance().getMainWorkflow();

  setWindowTitle("Application Host");
  this->setGeometry(150, 150, 1200, 600);

  newObjectDialog = new NewObjectDialog();

  tabWidget = new QTabWidget();
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
  connect(workflow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));

  toolBox = new QTreeWidget();
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

  fileMenu->addAction("Open Workflow", this, SLOT(loadWorkflow()), QKeySequence(Qt::CTRL + Qt::Key_O));
  fileMenu->insertSeparator(fileMenu->actions().last());
  fileMenu->addAction("Save Workflow", this, SLOT(saveWorkflow()), QKeySequence(Qt::CTRL + Qt::Key_S));
  fileMenu->addAction("Reload Workflow", this, SLOT(reload()), QKeySequence(Qt::CTRL + Qt::Key_R));

  fileMenu->addAction("Quit", this, SLOT(quit()));
  fileMenu->insertSeparator(fileMenu->actions().last());

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
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::itemClickedHandler(QTreeWidgetItem *item, int column) {
  if (item->childCount()) {
    item->setExpanded(!item->isExpanded());
  }
}

void MainWindow::itemDoubleClickedHandler(QTreeWidgetItem *item, int column) {
  if (!item->childCount()) {
    // new tool
    string classname = item->text(0).toUtf8().data();
    while ((item = item->parent())) { 
      classname = string(item->text(0).toUtf8().data()) + "::" + classname;
    }
    //cout << "New item: " << classname.c_str() << endl;
    DataModel::getInstance().getMainWorkflow()->newModule(classname);
  }
}

void MainWindow::newItem() {
  newObjectDialog->updateList();
  if (newObjectDialog->exec() == QDialog::Accepted && newObjectDialog->getSelectedClass().size()) {
    DataModel::getInstance().getMainWorkflow()->newModule(newObjectDialog->getSelectedClass().toUtf8().data());
  }
}

void MainWindow::loadWorkflow() {
  QFileDialog fileDialog(this);
  DataModel& model = DataModel::getInstance();

  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      Workflow* workflow = dynamic_cast<Workflow*>(Xmlizer::CreateReflectableClass(filenames[0].toUtf8().data()));
      if (workflow) {
        workflow->resumeFromModel();

        Workflow* oldWorkflow = model.getMainWorkflow();
        delete oldWorkflow;
        tabWidget->removeTab(0);
        model.setMainWorkflow(workflow);
        tabWidget->addTab(workflow->dispenseWidget(), "Main");
        connect(workflow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
        updateToolBox(toolBox);
      }
    }
  }
}

void MainWindow::saveWorkflow() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      Controller::getInstance().saveCurrentWorkflow(filenames[0].toUtf8().data());
    }
  }
}

void MainWindow::loadLibrary() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      Workflow* workflow = DataModel::getInstance().getMainWorkflow();
      vector<string>* libs = workflow->getLibraries();
      libs->push_back(filenames[0].toUtf8().data());
      workflow->setLibraries(libs);
    }
    reload();
  }
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  Workflow* workflow = model.getMainWorkflow();

  TiXmlElement* workflowElement = workflow->getXml(false);
  Xmlizer::ToXml(*workflowElement, *workflow);
  reflection::ReflectableClassFactory::getInstance().deleteInstance(workflow);
  workflow = 0;
  tabWidget->removeTab(0);
  workflow = dynamic_cast<Workflow*>(Xmlizer::CreateReflectableClass(*workflowElement));
  if (!workflow)
    throw "could not reload workflow";

  model.setMainWorkflow(workflow);
  workflow->resumeFromModel();
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
  connect(workflow, SIGNAL(updateFinished()), this, SLOT(updateFinished()));
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

void MainWindow::updateCurrentModule() {
  fileMenu->setEnabled(false);
  centralWidget()->setEnabled(false);

  DataModel::getInstance().getMainWorkflow()->updateSelectedModule();
}

void MainWindow::updateWorkflow() {
  fileMenu->setEnabled(false);
  centralWidget()->setEnabled(false);

  DataModel::getInstance().getMainWorkflow()->updateOutputs();
}

void MainWindow::terminateUpdate() {
  fileMenu->setEnabled(true);
  centralWidget()->setEnabled(true);
}

void MainWindow::updateFinished() {
  //cout << "Release UI" << endl;
  fileMenu->setEnabled(true);
  centralWidget()->setEnabled(true);
}

}

}
