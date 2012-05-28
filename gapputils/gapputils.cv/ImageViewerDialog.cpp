/*
 * ImageViewerDialog.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewerDialog.h"
#include <QWheelEvent>
#include <QMouseEvent>

#include <cmath>

namespace gapputils {

namespace cv {

ImageViewerWidget::ImageViewerWidget(int width, int height) : QGraphicsView(), viewScale(1.0) {
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  scene->setSceneRect(0, 0, width, height);
  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));

  //scene->addItem(new RectangleItem(model, this));
}

ImageViewerWidget::~ImageViewerWidget() {
}

void ImageViewerWidget::updateSize(int width, int height) {
  scene()->setSceneRect(0, 0, width, height);
}

void ImageViewerWidget::setBackgroundImage(boost::shared_ptr<QImage> image) {
  backgroundImage = image;
  setCacheMode(CacheNone);
  update();
  setCacheMode(CacheBackground);
}

void ImageViewerWidget::drawBackground(QPainter *painter, const QRectF &rect) {
  // Shadow
  const QRectF& sceneRect = this->sceneRect();

  if (backgroundImage) {
    painter->drawImage(0, 0, *backgroundImage);
  } else {
    // Fill
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(1, QColor(160, 160, 196));
    //painter->fillRect(sceneRect, Qt::white);
    painter->fillRect(sceneRect, gradient);
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(sceneRect);
  }

  QGraphicsView::drawBackground(painter, rect);
}

void ImageViewerWidget::scaleView(qreal scaleFactor)
{
  viewScale *= scaleFactor;
  qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
  if (factor < 0.07 || factor > 100)
    return;

  scale(scaleFactor, scaleFactor);
}

void ImageViewerWidget::mousePressEvent(QMouseEvent* event) {


  if (event->button() == Qt::LeftButton)
    setDragMode(ScrollHandDrag);
  else if (event->button() == Qt::RightButton)
    setDragMode(RubberBandDrag);

  QGraphicsView::mousePressEvent(event);
}

void ImageViewerWidget::mouseReleaseEvent(QMouseEvent* event) {
  QGraphicsView::mouseReleaseEvent(event);
  setDragMode(NoDrag);
}

void ImageViewerWidget::wheelEvent(QWheelEvent *event)
{
  scaleView(pow((double)1.3, event->delta() / 240.0));
}

qreal ImageViewerWidget::getViewScale() {
  return viewScale;
}

ImageViewerDialog::ImageViewerDialog(QWidget* widget) : QDialog(), widget(widget) {
  setGeometry(50, 50, widget->width(), widget->height());
  widget->setParent(this);
}

ImageViewerDialog::~ImageViewerDialog() {
}

void ImageViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

QWidget* ImageViewerDialog::getWidget() const {
  return widget;
}

}

}
