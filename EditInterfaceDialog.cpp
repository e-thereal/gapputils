/*
 * EditInterfaceDialog.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: tombr
 */

#include "EditInterfaceDialog.h"

#include <qgroupbox.h>
#include <qformlayout.h>
#include <qlabel.h>
#include <qlistwidget.h>
#include <qdialogbuttonbox.h>
#include <sstream>
#include <iostream>

namespace gapputils {

namespace host {

//  QGridLayout* layout = new QGridLayout();
//
//  layout->addWidget(new QLabel("Name:"), 0, 0);
//  layout->addWidget(new QLineEdit(), 0, 1);
//  layout->addWidget(new QLabel("Type:"), 1, 0);
//  layout->addWidget(new QLineEdit(), 1, 1);
//  layout->addWidget(new QLabel("Attributes:"), 2, 0, 1, 1, Qt::AlignTop);
//  layout->addWidget(attributesEdit = new QTextEdit(), 2, 1);

EditInterfaceDialog::EditInterfaceDialog(InterfaceDescription* interface, QWidget *parent)
 : QDialog(parent), interface(interface), currentProperty(0), attributesChangeable(true)
{
  setGeometry(x(), y(), 500, 400);

  isCombinerCB = new QCheckBox("Combiner Interface");
  isCombinerCB->setChecked(interface->getIsCombinerInterface());
  connect(isCombinerCB, SIGNAL(stateChanged(int)), this, SLOT(combinerStatusChanged(int)));

  QGroupBox* includesGB = new QGroupBox("Includes");
  QVBoxLayout* includesLayout = new QVBoxLayout();
  includesLayout->addWidget(includesEdit = new QTextEdit());
  includesGB->setLayout(includesLayout);
  includesEdit->clear();
  for(unsigned i = 0; i < interface->getHeaders()->size(); ++i) {
    includesEdit->append(interface->getHeaders()->at(i).c_str());
  }
  connect(includesEdit, SIGNAL(textChanged()), this, SLOT(includesChanged()));

  QGroupBox* propertiesGB = new QGroupBox("Properties");
  QVBoxLayout* propLayout = new QVBoxLayout();
  propLayout->addWidget(propertyList = new QListWidget());
  for(unsigned i = 0; i < interface->getPropertyDescriptions()->size(); ++i)
    propertyList->addItem(interface->getPropertyDescriptions()->at(i)->getName().c_str());
  propertyList->addItem("<New Property>");
  propertiesGB->setLayout(propLayout);
  connect(propertyList, SIGNAL(itemSelectionChanged()), this, SLOT(propertySelectionChanged()));

  QGroupBox* settingsGB = new QGroupBox("Settings");
  QFormLayout* layout = new QFormLayout();
  layout->addRow("Name:", nameEdit = new QLineEdit());
  layout->addRow("Type:", typeEdit = new QLineEdit());
  layout->addRow("Default:", defaultEdit = new QLineEdit());
  QVBoxLayout* attributesLayout = new QVBoxLayout;
  attributesLayout->addWidget(attributesEdit = new QTextEdit);
  layout->addRow("Attributes:", attributesLayout);
  settingsGB->setLayout(layout);
  connect(nameEdit, SIGNAL(editingFinished()), this, SLOT(nameChanged()));
  connect(typeEdit, SIGNAL(editingFinished()), this, SLOT(typeChanged()));
  connect(defaultEdit, SIGNAL(editingFinished()), this, SLOT(defaultChanged()));
  connect(attributesEdit, SIGNAL(textChanged()), this, SLOT(attributesChanged()));

  QHBoxLayout* propSetLayout = new QHBoxLayout();
  propSetLayout->addWidget(propertiesGB);
  propSetLayout->addWidget(settingsGB);

  QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(isCombinerCB);
  mainLayout->addWidget(includesGB);
  mainLayout->addLayout(propSetLayout);
  mainLayout->addWidget(buttonBox);
  setLayout(mainLayout);
  setWindowTitle("Check properties");

  deleteSC = new QShortcut(QKeySequence(Qt::Key_Delete), propertyList);
  connect(deleteSC, SIGNAL(activated()), this, SLOT(deleteItem()));
}

EditInterfaceDialog::~EditInterfaceDialog() {
  delete deleteSC;
}

void EditInterfaceDialog::propertySelectionChanged() {
  if (propertyList->selectedItems().empty())
    return;

  nameEdit->setText(propertyList->selectedItems()[0]->text());
  currentProperty = 0;
  for (unsigned i = 0; i < interface->getPropertyDescriptions()->size(); ++i) {
    if (interface->getPropertyDescriptions()->at(i)->getName().compare(propertyList->selectedItems()[0]->text().toAscii().data()) == 0) {
      currentProperty = interface->getPropertyDescriptions()->at(i).get();
      break;
    }
  }
  if (!currentProperty) {
    boost::shared_ptr<PropertyDescription> newProperty(new PropertyDescription());
    interface->getPropertyDescriptions()->push_back(newProperty);
    std::stringstream newPropName;
    newPropName << "NewProperty" << interface->getPropertyDescriptions()->size();
    newProperty->setName(newPropName.str());
    currentProperty = newProperty.get();
    propertyList->addItem("<New Property>");
  }
  nameEdit->setText(currentProperty->getName().c_str());
  typeEdit->setText(currentProperty->getType().c_str());
  defaultEdit->setText(currentProperty->getDefaultValue().c_str());
  attributesChangeable = false;
  attributesEdit->clear();
  for(unsigned i = 0; i < currentProperty->getPropertyAttributes()->size(); ++i) {
    attributesEdit->append(currentProperty->getPropertyAttributes()->at(i).c_str());
  }
  attributesChangeable = true;
  nameChanged();
}

void EditInterfaceDialog::nameChanged() {
  if (propertyList->selectedItems().empty() || !currentProperty)
    return;

  currentProperty->setName(nameEdit->text().toAscii().data());
  propertyList->selectedItems()[0]->setText(nameEdit->text());
}

void EditInterfaceDialog::typeChanged() {
  if (propertyList->selectedItems().empty() || !currentProperty)
    return;
  currentProperty->setType(typeEdit->text().toAscii().data());
}

void EditInterfaceDialog::defaultChanged() {
  if (propertyList->selectedItems().empty() || !currentProperty)
    return;
  currentProperty->setDefaultValue(defaultEdit->text().toAscii().data());
}

void EditInterfaceDialog::attributesChanged() {
  if (propertyList->selectedItems().empty() || !currentProperty || !attributesChangeable)
    return;
  std::stringstream stream(attributesEdit->document()->toPlainText().toAscii().data());
  std::string line;
  currentProperty->getPropertyAttributes()->clear();
  while(std::getline(stream, line)) {
    if (line.length())
      currentProperty->getPropertyAttributes()->push_back(line);
  }
}

void EditInterfaceDialog::deleteItem() {
  currentProperty = 0;
  for (unsigned i = 0; i < interface->getPropertyDescriptions()->size(); ++i) {
    if (interface->getPropertyDescriptions()->at(i)->getName().compare(propertyList->selectedItems()[0]->text().toAscii().data()) == 0) {
      currentProperty = interface->getPropertyDescriptions()->at(i).get();
      interface->getPropertyDescriptions()->erase(interface->getPropertyDescriptions()->begin() + i);
      break;
    }
  }
  QListWidgetItem* item = propertyList->currentItem();
  if (propertyList->currentRow() + 2 == propertyList->count())
    propertyList->setCurrentRow(std::max(0, propertyList->currentRow() - 1));
  delete item;
}

void EditInterfaceDialog::includesChanged() {
  std::stringstream stream(includesEdit->document()->toPlainText().toAscii().data());
  std::string line;
  interface->getHeaders()->clear();
  while(std::getline(stream, line)) {
    if (line.length())
      interface->getHeaders()->push_back(line);
  }
}

void EditInterfaceDialog::combinerStatusChanged(int state) {
  interface->setIsCombinerInterface(isCombinerCB->isChecked());
}

}

}
