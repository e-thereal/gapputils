#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qfiledialog.h>

#include <Xmlizer.h>

#include "DataModel.h"

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

  fileMenu = menuBar()->addMenu("File");
  QAction* newItemAction = fileMenu->addAction("New Item");
  QAction* loadWFAction = fileMenu->addAction("Load Workflow");
  QAction* quitAction = fileMenu->addAction("Quit");

  tabWidget = new QTabWidget();

  Workflow* workflow = DataModel::getInstance().getMainWorkflow();
  tabWidget->addTab(workflow->getWidget(), "Root");
  setCentralWidget(tabWidget);

  connect(newItemAction, SIGNAL(triggered()), this, SLOT(newItem()));
  connect(loadWFAction, SIGNAL(triggered()), this, SLOT(loadWorkflow()));
  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

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

}

}
