/*
 * EditInterfaceDialog.h
 *
 *  Created on: Aug 18, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSHOST_EDITINTERFACEDIALOG_H_
#define GAPPUTILSHOST_EDITINTERFACEDIALOG_H_

#include <QDialog>
#include <qlistwidget.h>
#include <qlineedit.h>
#include <qtextedit.h>
#include <qshortcut.h>
#include <qcheckbox.h>

#include <gapputils/InterfaceDescription.h>

namespace gapputils {

namespace host {

class EditInterfaceDialog : public QDialog {
  Q_OBJECT

private:
  QCheckBox* isCombinerCB;
  QListWidget* propertyList;
  QLineEdit *nameEdit, *typeEdit, *defaultEdit;
  QTextEdit *attributesEdit, *includesEdit;
  InterfaceDescription* interface;
  PropertyDescription* currentProperty;
  QShortcut* deleteSC;

  bool attributesChangeable;

public:
  EditInterfaceDialog(InterfaceDescription* interface, QWidget *parent = 0);
  virtual ~EditInterfaceDialog();

private Q_SLOTS:
  void propertySelectionChanged();
  void nameChanged();
  void typeChanged();
  void defaultChanged();
  void attributesChanged();
  void deleteItem();
  void includesChanged();
  void combinerStatusChanged(int state);
};

}

}
#endif /* GAPPUTILSHOST_EDITINTERFACEDIALOG_H_ */
