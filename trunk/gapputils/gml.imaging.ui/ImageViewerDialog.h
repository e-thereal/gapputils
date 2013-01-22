/*
 * ImageViewerDialog.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GML_IMAGING_UI_IMAGEVIEWERDIALOG_H_
#define GML_IMAGING_UI_IMAGEVIEWERDIALOG_H_

#include <QDialog>
#include <QGraphicsView>
#include <qtimer.h>
#include <qimage.h>
#include <qlabel.h>

#include <boost/shared_ptr.hpp>

#include <gapputils/Image.h>

#include <capputils/Enumerators.h>
#include <capputils/ObservableClass.h>

namespace gml {

namespace imaging {

namespace ui {

class ImageViewer;
class ImageViewerDialog;

class ImageViewerWidget : public QGraphicsView {
  Q_OBJECT

private:
  ImageViewer* viewer;
  ImageViewerDialog* dialog;
  qreal viewScale;
  boost::shared_ptr<QTimer> timer;
  std::vector<boost::shared_ptr<gapputils::image_t> > images;
  QPointF dragStart;
  boost::shared_ptr<QImage> qimage;

public:
  ImageViewerWidget(ImageViewer* viewer, ImageViewerDialog* dialog);

  void scaleView(qreal scaleFactor);
  qreal getViewScale();

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void drawBackground(QPainter *painter, const QRectF &rect);
  void wheelEvent(QWheelEvent *event);
  void keyPressEvent(QKeyEvent *event);

  void changedHandler(capputils::ObservableClass* sender, int eventId);

public Q_SLOTS:
  void updateView();
};

class ImageViewerDialog : public QDialog {
  Q_OBJECT

private:
  boost::shared_ptr<ImageViewerWidget> widget;
  boost::shared_ptr<QLabel> helpLabel;

public:
  ImageViewerDialog(ImageViewer* viewer);

protected:
  void keyPressEvent(QKeyEvent *event);
  void keyReleaseEvent(QKeyEvent *event);
  virtual void resizeEvent(QResizeEvent* resizeEvent);
};

}

}

}

#endif /* GML_IMAGING_UI_IMAGEVIEWERDIALOG_H_ */
