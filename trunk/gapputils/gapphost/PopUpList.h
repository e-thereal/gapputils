/*
 * PopUpList.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_POPUPLIST_H_
#define GAPPHOST_POPUPLIST_H_

#include <qdialog.h>
#include <qlistwidget.h>

namespace gapputils {

namespace host {

class PopUpList : public QDialog {
  Q_OBJECT

private:
  QListWidget* list;

public:
  PopUpList(QString title = "Make a selection", QWidget* parent = 0);
  virtual ~PopUpList();

  QListWidget* getList();
};

}

}

#endif /* POPUPLIST_H_ */
