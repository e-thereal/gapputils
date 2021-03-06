/*
 * PopUpList.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "PopUpList.h"
#include <qboxlayout.h>
#include <qcursor.h>
#include <qlabel.h>
#include <qframe.h>
#include <qfontmetrics.h>

#include <qdesktopwidget.h>

namespace gapputils {

namespace host {

PopUpList::PopUpList(QString title, QWidget* parent) : QDialog(parent, Qt::Popup) {
  QLabel* label = new QLabel(title);

  list = new QListWidget();
  connect(list, SIGNAL(itemSelectionChanged()), this, SLOT(accept()));

  QVBoxLayout* layout = new QVBoxLayout(this);
  layout->setMargin(4);
  layout->addWidget(label);
  layout->addWidget(list);

  QFrame* frame = new QFrame();
  frame->setFrameStyle(QFrame::Panel | QFrame::Raised);
  frame->setLineWidth(2);
  frame->setLayout(layout);

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->setMargin(0);
  mainLayout->addWidget(frame);
  setLayout(mainLayout);

  QFontMetrics fontMetrics(label->font());
  const int width = std::max(200, fontMetrics.boundingRect(title).width() + 20), height = 200;

  QDesktopWidget desk;
  int screenWidth = 0, screenHeight = 0;

  for (int iScreen = 0; iScreen <= desk.screenNumber(QCursor::pos()); ++iScreen) {
    screenWidth  += desk.screenGeometry(iScreen).width();
    screenHeight += desk.screenGeometry(iScreen).height();
  }

  setGeometry(std::min(screenWidth - width - 10, QCursor::pos().x()),
      std::min(screenHeight - height - 10, QCursor::pos().y()),
      width, height);
}

PopUpList::~PopUpList() {
}

QListWidget* PopUpList::getList() {
  return list;
}

}

}
