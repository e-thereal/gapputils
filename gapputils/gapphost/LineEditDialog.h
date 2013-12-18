/*
 * LineEditDialog.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_LINEEDITDIALOG_H_
#define GAPPHOST_LINEEDITDIALOG_H_

#include <qdialog.h>
#include <qlineedit.h>

namespace gapputils {

namespace host {

class LineEditDialog : public QDialog {
  Q_OBJECT

private:
  QLineEdit* edit;

public:
  LineEditDialog(QString title, QWidget* parent);
  virtual ~LineEditDialog();

  QString getText();
};

}

}

#endif /* GAPPHOST_LINEEDITDIALOG_H_ */
