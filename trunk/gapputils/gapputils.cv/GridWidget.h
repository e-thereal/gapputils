#pragma once
#ifndef GAPPUTILSCV_GRIDWIDGET_H_
#define GAPPUTILSCV_GRIDWIDGET_H_

#include <QGraphicsView>
#include <QImage>

namespace gapputils {

namespace cv {

class GridModel;

class GridWidget : public QGraphicsView
{
  Q_OBJECT

private:
  GridModel* model;
  QImage* backgroundImage;
  qreal viewScale;

public:
  GridWidget(GridModel* model, int width, int height, QWidget* parent = 0);
  virtual ~GridWidget(void);

  void resumeFromModel();
  void renewGrid(int rowCount, int columnCount);
  void updateSize(int width, int height);

  void setBackgroundImage(QImage* image);
  void scaleView(qreal scaleFactor);

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void drawBackground(QPainter *painter, const QRectF &rect);
  void wheelEvent(QWheelEvent *event);
};

}

}

#endif /* GAPPUTILSCV_GRIDWIDGET_H_ */
