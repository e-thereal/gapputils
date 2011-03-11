#include "NewObjectDialog.h"

#include <ReflectableClassFactory.h>

using namespace std;
using namespace capputils::reflection;

NewObjectDialog::NewObjectDialog(QWidget *parent)
    : QDialog(parent)
{
  ui.setupUi(this);
  //setFixedSize(381, 311);

  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  vector<string>& classNames = factory.getClassNames();
  for (unsigned i = 0; i < classNames.size(); ++i)
    ui.listWidget->addItem(classNames[i].c_str());

  connect(ui.cancelButton, SIGNAL(clicked(bool)), this, SLOT(cancelButtonClicked(bool)));
  connect(ui.addButton, SIGNAL(clicked(bool)), this, SLOT(addButtonClicked(bool)));
  connect(ui.listWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(doubleClickedHandler(QListWidgetItem*)));
}

NewObjectDialog::~NewObjectDialog()
{
}

QString NewObjectDialog::getSelectedClass() const {
  if (ui.listWidget->selectedItems().size())
    return ui.listWidget->selectedItems()[0]->text();
  return "";
}

void NewObjectDialog::cancelButtonClicked(bool) {
  reject();
}

void NewObjectDialog::addButtonClicked(bool) {
  accept();
}

void NewObjectDialog::doubleClickedHandler(QListWidgetItem* /*item*/) {
  accept();
}
