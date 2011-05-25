#include "NewObjectDialog.h"

#include <capputils/ReflectableClassFactory.h>
#include <algorithm>

using namespace std;
using namespace capputils::reflection;

NewObjectDialog::NewObjectDialog(QWidget *parent)
    : QDialog(parent)
{
  ui.setupUi(this);
  //setFixedSize(381, 311);

  updateList();

  connect(ui.cancelButton, SIGNAL(clicked(bool)), this, SLOT(cancelButtonClicked(bool)));
  connect(ui.addButton, SIGNAL(clicked(bool)), this, SLOT(addButtonClicked(bool)));
  connect(ui.listWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(doubleClickedHandler(QListWidgetItem*)));
}

NewObjectDialog::~NewObjectDialog()
{
}

void NewObjectDialog::updateList() {
  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  vector<string> classNames = factory.getClassNames();
  sort(classNames.begin(), classNames.end());
  ui.listWidget->clear();
  for (unsigned i = 0; i < classNames.size(); ++i)
    ui.listWidget->addItem(classNames[i].c_str());
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
