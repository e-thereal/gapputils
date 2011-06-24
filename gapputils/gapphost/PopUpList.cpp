/*
 * PopUpList.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "PopUpList.h"

namespace gapputils {

namespace host {

PopUpList::PopUpList(QWidget* parent) : QDialog(parent) {
  list = new QListWidget(this);
  list->setGeometry(1, 1, 150, 150);
  connect(list, SIGNAL(itemSelectionChanged()), this, SLOT(accept()));
}

PopUpList::~PopUpList() {
}

QListWidget* PopUpList::getList() {
  return list;
}

}

}
