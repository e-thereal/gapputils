/*
 * RectangleWidget.h
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_RECTANGLEWIDGET_H_
#define GAPPUTILSCV_RECTANGLEWIDGET_H_

#include <QGraphicsView>

#include <boost/shared_ptr.hpp>

#include "RectangleModel.h"

namespace gapputils {

namespace cv {

class RectangleWidget : public QGraphicsView {
  Q_OBJECT

private:
  boost::shared_ptr<QImage> backgroundImage;
  qreal viewScale;

public:
  RectangleWidget(int width, int height, boost::shared_ptr<RectangleModel> model);
  virtual ~RectangleWidget();

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

}

}

#endif /* GAPPUTILSCV_RECTANGLEWIDGET_H_ */
