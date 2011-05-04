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
  Workflow* workflow = DataModel::getInstance().getMainWorkflow();
  tabWidget->addTab(workflow->getWidget(), "Root");
  setCentralWidget(tabWidget);

  fileMenu = menuBar()->addMenu("File");
  connect(fileMenu->addAction("New Item"), SIGNAL(triggered()), this, SLOT(newItem()));
  connect(fileMenu->addAction("Load Workflow"), SIGNAL(triggered()), this, SLOT(loadWorkflow()));
  connect(fileMenu->addAction("Save Workflow"), SIGNAL(triggered()), this, SLOT(saveWorkflow()));
  connect(fileMenu->addAction("Load Library"), SIGNAL(triggered()), this, SLOT(loadLibrary()));
  connect(fileMenu->addAction("Quit"), SIGNAL(triggered()), this, SLOT(quit()));

  workflow->resumeFromModel();
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
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      Workflow* workflow = dynamic_cast<Workflow*>(Xmlizer::CreateReflectableClass(filenames[0].toUtf8().data()));
      if (workflow) {
        workflow->resumeFromModel();
        tabWidget->addTab(workflow->getWidget(), "New");
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
      string filename = filenames[0].toUtf8().data();
      LibraryLoader::getInstance().loadLibrary(filename);
      vector<string>* libs = DataModel::getInstance().getMainWorkflow()->getLibraries();
      libs->push_back(filename);
      DataModel::getInstance().getMainWorkflow()->setLibraries(libs);
    }
  }
}

}

}
