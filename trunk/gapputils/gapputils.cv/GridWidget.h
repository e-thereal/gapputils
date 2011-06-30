#pragma once
#ifndef GAPPUTILSCV_GRIDWIDGET_H_
#define GAPPUTILSCV_GRIDWIDGET_H_

#include <QGraphicsView>
#include <QImage>

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace cv {

class GridModel;

class GridWidget : public QGraphicsView
{
  Q_OBJECT

private:
  boost::shared_ptr<GridModel> model;
  boost::shared_ptr<QImage> backgroundImage;
  qreal viewScale;

public:
  GridWidget(boost::shared_ptr<GridModel> model, int width, int height, QWidget* parent = 0);
  virtual ~GridWidget(void);

  void resumeFromModel(boost::shared_ptr<GridModel> model);
  void renewGrid(int rowCount, int columnCount);
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

#endif /* GAPPUTILSCV_GRIDWIDGET_H_ */
