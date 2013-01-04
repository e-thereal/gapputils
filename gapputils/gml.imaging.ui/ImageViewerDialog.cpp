/*
 * ImageViewerDialog.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewerDialog.h"
#include <QWheelEvent>
#include <QMouseEvent>
#include <qimage.h>

#include <cmath>

using namespace gapputils;

namespace gml {

namespace imaging {

namespace ui {

ImageViewerWidget::ImageViewerWidget() : QGraphicsView(), viewScale(1.0) {
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  scene->setSceneRect(0, 0, 128, 128);
  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));
}

ImageViewerWidget::~ImageViewerWidget() {
}

void ImageViewerWidget::setBackgroundImage(boost::shared_ptr<image_t> image) {
  scene()->setSceneRect(0, 0, image->getSize()[0], image->getSize()[1]);
  backgroundImage = image;
  setCacheMode(CacheNone);
  update();
  setCacheMode(CacheBackground);
}

#define F_TO_INT(value) std::min(255, std::max(0, (int)(value * 256)))

void ImageViewerWidget::drawBackground(QPainter *painter, const QRectF &rect) {
  // Shadow
  const QRectF& sceneRect = this->sceneRect();

  if (backgroundImage) {
    const int width = backgroundImage->getSize()[0];
    const int height = backgroundImage->getSize()[1];
    const int depth = backgroundImage->getSize()[2];

    const int count = width * height;
    QImage image(width, height, QImage::Format_ARGB32);

    float* buffer = backgroundImage->getData();

    for (int i = 0, y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {
        int r = F_TO_INT(buffer[i]);
        int g = depth == 3 ? F_TO_INT(buffer[i + count]) : r;
        int b = depth == 3 ? F_TO_INT(buffer[i + 2 * count]) : r;
        image.setPixel(x, y, QColor(r, g, b).rgb());
      }
    }

    painter->drawImage(0, 0, image);
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

ImageViewerDialog::ImageViewerDialog() : QDialog(), widget(new ImageViewerWidget()) {
  setGeometry(50, 50, widget->width(), widget->height());
  widget->setParent(this);
}

ImageViewerDialog::~ImageViewerDialog() {
}

void ImageViewerDialog::setBackgroundImage(boost::shared_ptr<image_t> image) {
  widget->setBackgroundImage(image);
}

void ImageViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

}

}

}
