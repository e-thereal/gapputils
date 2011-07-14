/*
 * RectangleDialog.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RECTANGLEDIALOG_H_
#define GAPPUTILSCV_RECTANGLEDIALOG_H_

#include <QDialog>

namespace gapputils {

namespace cv {

class RectangleDialog : public QDialog {
  Q_OBJECT

private:
  QWidget* widget;

public:
  RectangleDialog(QWidget* widget);
  virtual ~RectangleDialog();

  virtual void resizeEvent(QResizeEvent* resizeEvent);

  QWidget* getWidget() const;
};

}

}

#endif /* GAPPUTILSCV_RECTANGLEDIALOG_H_ */
