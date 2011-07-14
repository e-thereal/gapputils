/*
 * RectangleDialog.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "RectangleDialog.h"

namespace gapputils {

namespace cv {

RectangleDialog::RectangleDialog(QWidget* widget) : QDialog(), widget(widget) {
  setGeometry(50, 50, widget->width(), widget->height());
  widget->setParent(this);
}

RectangleDialog::~RectangleDialog() {
}

void RectangleDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

QWidget* RectangleDialog::getWidget() const {
  return widget;
}

}

}
