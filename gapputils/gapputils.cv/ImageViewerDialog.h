/*
 * ImageViewerDialog.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_IMAGEVIEWERDIALOG_H_
#define GAPPUTILSCV_IMAGEVIEWERDIALOG_H_

#include <QDialog>
#include <QGraphicsView>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace cv {

class ImageViewerWidget : public QGraphicsView {
  Q_OBJECT

private:
  boost::shared_ptr<QImage> backgroundImage;
  qreal viewScale;

public:
  ImageViewerWidget(int width, int height);
  virtual ~ImageViewerWidget();

  void updateSize(int width, int height);
  void setBackgroundImage(boost::shared_ptr<QImage> image);
  void scaleView(qreal scaleFactor);
  qreal getViewScale();

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void drawBackground(QPainter *painter, const QRectF &rect);
  void wheelEvent(QWheelEvent *event);
};

class ImageViewerDialog : public QDialog {
  Q_OBJECT

private:
  QWidget* widget;

public:
  ImageViewerDialog(QWidget* widget);
  virtual ~ImageViewerDialog();

  virtual void resizeEvent(QResizeEvent* resizeEvent);

  QWidget* getWidget() const;
};

}

}

#endif /* GAPPUTILSCV_IMAGEVIEWERDIALOG_H_ */
