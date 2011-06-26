#pragma once
#ifndef GAPPUTILSCV_GRIDDIALOG_H_
#define GAPPUTILSCV_GRIDDIALOG_H_

#include <QDialog>
#include <QImage>

#include "GridWidget.h"

namespace gapputils {

namespace cv {

class GridModel;

class GridDialog : public QDialog
{
  Q_OBJECT

private:
  GridWidget* gridWidget;

public:
  GridDialog(GridModel* model, int width, int height);
  virtual ~GridDialog(void);

  void renewGrid(int rowCount, int columnCount);
  void updateSize(int width, int height);
  void setBackgroundImage(QImage* image);

  virtual void resizeEvent(QResizeEvent* resizeEvent);

};

}

}

#endif /* GAPPUTILSCV_GRIDDIALOG_H_ */