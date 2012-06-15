/*
 * MakeGlobalDialog.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_MAKEGLOBALDIALOG_H_
#define GAPPHOST_MAKEGLOBALDIALOG_H_

#include <qdialog.h>
#include <qlineedit.h>

namespace gapputils {

namespace host {

class MakeGlobalDialog : public QDialog {
  Q_OBJECT

private:
  QLineEdit* edit;

public:
  MakeGlobalDialog(QWidget* parent);
  virtual ~MakeGlobalDialog();

  QString getText();
};

}

}

#endif /* GAPPHOST_MAKEGLOBALDIALOG_H_ */
