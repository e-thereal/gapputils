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

#include "ImageViewer.h"

#include <capputils/EventHandler.h>

#include <cmath>

using namespace gapputils;

namespace gml {

namespace imaging {

namespace ui {

ImageViewerWidget::ImageViewerWidget(ImageViewer* viewer) : QGraphicsView(), viewer(viewer), viewScale(1.0) {
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  if (viewer->getBackgroundImage()) {
    image_t& image = *viewer->getBackgroundImage();
    scene->setSceneRect(0, 0, image.getSize()[0], image.getSize()[1]);
  } else {
    scene->setSceneRect(0, 0, 128, 128);
  }

  if (viewer->getMode() == ViewMode::Wobble) {
    timer = boost::make_shared<QTimer>(this);
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(updateView()));
    timer->start(viewer->getWobbleDelay());
  }

  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));

  viewer->Changed.connect(capputils::EventHandler<ImageViewerWidget>(this, &ImageViewerWidget::changedHandler));
}

void ImageViewerWidget::updateView() {
  if (viewer->getBackgroundImage()) {
    image_t& image = *viewer->getBackgroundImage();
    scene()->setSceneRect(0, 0, image.getSize()[0], image.getSize()[1]);
  }
  setCacheMode(CacheNone);
  update();
  setCacheMode(CacheBackground);
}

#define F_TO_INT(value) std::min(255, std::max(0, (int)(value * 256)))

void ImageViewerWidget::drawBackground(QPainter *painter, const QRectF &rect) {
  static int updateCounter = 0;

  // Shadow
  const QRectF& sceneRect = this->sceneRect();

  if (viewer->getBackgroundImage()) {
    ++updateCounter;
    std::cout << "Drawing: " << updateCounter << std::endl;

    image_t& backgroundImage = *viewer->getBackgroundImage();
    const int width = backgroundImage.getSize()[0];
    const int height = backgroundImage.getSize()[1];
    const int depth = backgroundImage.getSize()[2];

    const int count = width * height;
    QImage image(width, height, QImage::Format_ARGB32);

    float* buffer = backgroundImage.getData();

    switch (viewer->getMode()) {
    case ViewMode::Default:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int r = F_TO_INT(buffer[i]);
          int g = depth == 3 ? F_TO_INT(buffer[i + count]) : r;
          int b = depth == 3 ? F_TO_INT(buffer[i + 2 * count]) : r;
          image.setPixel(x, y, QColor(r, g, b).rgb());
        }
      }
      break;

    case ViewMode::Wobble:
      {
        int iSlice = (updateCounter % 2) % depth;
        for (int i = 0, y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x, ++i) {
            int r = F_TO_INT(buffer[i + iSlice * count]);
            int g = F_TO_INT(buffer[i + iSlice * count]);
            int b = F_TO_INT(buffer[i + iSlice * count]);
            image.setPixel(x, y, QColor(r, g, b).rgb());
          }
        }
      }
      break;
    }

    painter->drawImage(0, 0, image);
  } else {
    // Fill
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(1, QColor(160, 160, 196));
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


void ImageViewerWidget::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {

  if (eventId == ImageViewer::modeId) {
    if (viewer->getMode() == ViewMode::Wobble) {
      timer = boost::make_shared<QTimer>(this);
      connect(timer.get(), SIGNAL(timeout()), this, SLOT(updateView()));
      timer->start(viewer->getWobbleDelay());
    } else {
      timer = boost::shared_ptr<QTimer>();
    }
    updateView();
  }

  if (eventId == ImageViewer::backgroundId) {
    updateView();
  }
}

ImageViewerDialog::ImageViewerDialog(ImageViewer* viewer) : QDialog(), widget(new ImageViewerWidget(viewer)) {
  setGeometry(50, 50, widget->width(), widget->height());
  widget->setParent(this);
}

void ImageViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

}

}

}
