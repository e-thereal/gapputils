#pragma once
#ifndef GAPPUTILSCV_GRIDDIALOG_H_
#define GAPPUTILSCV_GRIDDIALOG_H_

#include <QDialog>
#include <QImage>

#include <boost/shared_ptr.hpp>

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
  GridDialog(boost::shared_ptr<GridModel> model, int width, int height);
  virtual ~GridDialog(void);

  void renewGrid(int rowCount, int columnCount);
  void updateSize(int width, int height);
  void setBackgroundImage(boost::shared_ptr<QImage> image);
  void resumeFromModel(boost::shared_ptr<GridModel> model);

  virtual void resizeEvent(QResizeEvent* resizeEvent);

};

}

}

#endif /* GAPPUTILSCV_GRIDDIALOG_H_ */
