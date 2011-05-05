#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qfiledialog.h>

#include <Xmlizer.h>
#include <LibraryLoader.h>

#include "DataModel.h"
#include "Controller.h"

using namespace std;
using namespace capputils;

namespace gapputils {

using namespace workflow;

namespace host {

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
  setWindowTitle("Application Host");
  this->setGeometry(150, 150, 800, 600);

  newObjectDialog = new NewObjectDialog();

  tabWidget = new QTabWidget();
  tabWidget->addTab(DataModel::getInstance().getMainWorkflow()->dispenseWidget(), "Main");
  setCentralWidget(tabWidget);

  fileMenu = menuBar()->addMenu("File");
  connect(fileMenu->addAction("New Item"), SIGNAL(triggered()), this, SLOT(newItem()));
  connect(fileMenu->addAction("Load Workflow"), SIGNAL(triggered()), this, SLOT(loadWorkflow()));
  connect(fileMenu->addAction("Save Workflow"), SIGNAL(triggered()), this, SLOT(saveWorkflow()));
  connect(fileMenu->addAction("Reload Workflow"), SIGNAL(triggered()), this, SLOT(reload()));
  connect(fileMenu->addAction("Load Library"), SIGNAL(triggered()), this, SLOT(loadLibrary()));
  connect(fileMenu->addAction("Quit"), SIGNAL(triggered()), this, SLOT(quit()));
}

MainWindow::~MainWindow()
{
  delete fileMenu;
}

void MainWindow::quit() {
  this->close();
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
  }
}

void MainWindow::reload() {
  DataModel& model = DataModel::getInstance();
  Workflow* workflow = model.getMainWorkflow();

  TiXmlElement* workflowElement = workflow->getXml(false);
  Xmlizer::ToXml(*workflowElement, *workflow);
  delete workflow;
  workflow = 0;
  tabWidget->removeTab(0);
  workflow = dynamic_cast<Workflow*>(Xmlizer::CreateReflectableClass(*workflowElement));
  if (!workflow)
    throw "could not reload workflow";

  model.setMainWorkflow(workflow);
  workflow->resumeFromModel();
  tabWidget->addTab(workflow->dispenseWidget(), "Main");
}

}

}
