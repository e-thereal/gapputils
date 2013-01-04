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

#include <boost/shared_ptr.hpp>

#include <gapputils/Image.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace ui {

class ImageViewerWidget : public QGraphicsView {
  Q_OBJECT

private:
  boost::shared_ptr<gapputils::image_t> backgroundImage;
  qreal viewScale;

public:
  ImageViewerWidget();
  virtual ~ImageViewerWidget();

  void setBackgroundImage(boost::shared_ptr<gapputils::image_t> image);
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
  boost::shared_ptr<ImageViewerWidget> widget;

public:
  ImageViewerDialog();
  virtual ~ImageViewerDialog();

  void setBackgroundImage(boost::shared_ptr<gapputils::image_t> image);

  virtual void resizeEvent(QResizeEvent* resizeEvent);
};

}

}

}

#endif /* GML_IMAGING_UI_IMAGEVIEWERDIALOG_H_ */
