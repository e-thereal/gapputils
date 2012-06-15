/*
 * MakeGlobalDialog.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "MakeGlobalDialog.h"



namespace gapputils {

namespace host {

MakeGlobalDialog::MakeGlobalDialog(QWidget* parent) : QDialog(parent) {
  edit = new QLineEdit(this);
  edit->setGeometry(1, 1, 150, 23);
  connect(edit, SIGNAL(editingFinished()), this, SLOT(accept()));
}

MakeGlobalDialog::~MakeGlobalDialog() {
}

QString MakeGlobalDialog::getText() {
  return edit->text();
}

}

}
