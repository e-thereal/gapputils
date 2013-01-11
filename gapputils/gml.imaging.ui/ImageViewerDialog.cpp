/*
 * ImageViewerDialog.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewerDialog.h"
#include <QWheelEvent>
#include <QMouseEvent>

#include "ImageViewer.h"

#include <capputils/EventHandler.h>

#include <sstream>
#include <cmath>

using namespace gapputils;

namespace gml {

namespace imaging {

namespace ui {

ImageViewerWidget::ImageViewerWidget(ImageViewer* viewer, ImageViewerDialog* dialog)
 : QGraphicsView(), viewer(viewer), dialog(dialog), viewScale(1.0)
{
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  if (viewer->getImage()) {
    image_t& image = *viewer->getImage();
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

  images.clear();
  if (viewer->getImage())
    images.push_back(viewer->getImage());
  if (viewer->getImages()) {
    for (size_t i = 0; i < viewer->getImages()->size(); ++i)
      images.push_back(viewer->getImages()->at(i));
  }

  viewer->setCurrentImage(viewer->getCurrentImage());
}

#define F_TO_INT(value) std::min(255, std::max(0, (int)(value * 256)))

void ImageViewerWidget::updateView() {
  std::stringstream title;
  title << viewer->getLabel() << " (Image: ";
  if (images.size())
    title << viewer->getCurrentImage() + 1 << "/" << images.size();
  else
    title << "0/0";
  title << ";" << "Slice: ";
  if (images.size() && images[viewer->getCurrentImage()]->getSize()[2])
    title << viewer->getCurrentSlice() + 1 << "/" << images[viewer->getCurrentImage()]->getSize()[2];
  else
    title << "0/0";
  title << ")";
  dialog->setWindowTitle(title.str().c_str());

  if (images.size()) {
    image_t& image = *images[viewer->getCurrentImage()];
    scene()->setSceneRect(0, 0, image.getSize()[0], image.getSize()[1]);

    const int width = image.getSize()[0];
    const int height = image.getSize()[1];
    const int depth = image.getSize()[2];

    const int count = width * height;
    qimage = boost::make_shared<QImage>(width, height, QImage::Format_ARGB32);

    float* buffer = image.getData();

    switch (viewer->getMode()) {
    case ViewMode::Default:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int r = F_TO_INT(buffer[i]);
          int g = depth == 3 ? F_TO_INT(buffer[i + count]) : r;
          int b = depth == 3 ? F_TO_INT(buffer[i + 2 * count]) : r;
          qimage->setPixel(x, y, QColor(r, g, b).rgb());
        }
      }
      break;

    case ViewMode::Wobble:
      {
        static int updateCounter = 0;
        int iSlice = (++updateCounter % 2) % depth;
        for (int i = 0, y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x, ++i) {
            int c = F_TO_INT(buffer[i + iSlice * count]);
            qimage->setPixel(x, y, QColor(c, c, c).rgb());
          }
        }
      }
      break;

    case ViewMode::Volume:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int c = F_TO_INT((buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          qimage->setPixel(x, y, QColor(c, c, c).rgb());
        }
      }
      break;
    }
  }
  setCacheMode(CacheNone);
  update();
  setCacheMode(CacheBackground);
}

void ImageViewerWidget::drawBackground(QPainter *painter, const QRectF &rect) {
  if (qimage) {
    painter->drawImage(0, 0, *qimage);
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

  dragStart = mapToScene(event->pos());

  QGraphicsView::mousePressEvent(event);
}

void ImageViewerWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() == Qt::RightButton) {
    QPointF dragEnd = mapToScene(event->pos());
    if (images.size()) {
      image_t& image = *images[viewer->getCurrentImage()];

      const int width = image.getSize()[0], height = image.getSize()[1], count = width * height;

      const int rx = std::max(0, std::min(width - 1, (int)std::min(dragStart.x(), dragEnd.x())));
      const int ry = std::max(0, std::min(height - 1, (int)std::min(dragStart.y(), dragEnd.y())));
      const int rwidth = std::max(0, std::min(width, (int)std::max(dragStart.x(), dragEnd.x()) - rx));
      const int rheight = std::max(0, std::min(height, (int)std::max(dragStart.y(), dragEnd.y()) - ry));

      float* buffer = image.getData();

      float minimum, maximum;
      minimum = maximum = buffer[viewer->getCurrentSlice() * count + ry * width + rx];

      for (int y = ry; y < ry + rheight; ++y) {
        for (int x = rx; x < rx + rwidth; ++x) {
          minimum = std::min(minimum, buffer[viewer->getCurrentSlice() * count + y * width + x]);
          maximum = std::max(maximum, buffer[viewer->getCurrentSlice() * count + y * width + x]);
        }
      }

      float contrastMargin = (maximum - minimum) * (1.0 - viewer->getContrast()) / 2;

      viewer->setMinimumIntensity(minimum - contrastMargin);
      viewer->setMaximumIntensity(maximum + contrastMargin);
    }
  }

  QGraphicsView::mouseReleaseEvent(event);
  setDragMode(NoDrag);
}

void ImageViewerWidget::wheelEvent(QWheelEvent *event) {
  scaleView(pow((double)1.3, event->delta() / 240.0));
}

void ImageViewerWidget::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_Q:
    dialog->close();
    break;

  case Qt::Key_W:
  {
    QRect rect = dialog->geometry();
    rect.setWidth(scene()->sceneRect().width() * viewScale + 5);
    rect.setHeight(scene()->sceneRect().height() * viewScale + 5);
    dialog->setGeometry(rect);
  }
    break;

  case Qt::Key_R:
    scaleView(std::min((qreal)(dialog->geometry().width() - 5.0) / (qreal)scene()->sceneRect().width(),
        (qreal)(dialog->geometry().height() - 5.0) / (qreal)scene()->sceneRect().height()) / viewScale);
    break;

  case Qt::Key_Left:
  case Qt::Key_S:
    viewer->setCurrentImage(viewer->getCurrentImage() - 1);
    break;

  case Qt::Key_Right:
  case Qt::Key_F:
    viewer->setCurrentImage(viewer->getCurrentImage() + 1);
    break;

  case Qt::Key_Up:
  case Qt::Key_E:
    viewer->setCurrentSlice(viewer->getCurrentSlice() + 1);
    break;

  case Qt::Key_Down:
  case Qt::Key_D:
    viewer->setCurrentSlice(viewer->getCurrentSlice() - 1);
    break;
  }
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

  if (eventId == ImageViewer::imageId || eventId == ImageViewer::imagesId) {
    images.clear();
    if (viewer->getImage())
      images.push_back(viewer->getImage());
    if (viewer->getImages()) {
      for (size_t i = 0; i < viewer->getImages()->size(); ++i)
        images.push_back(viewer->getImages()->at(i));
    }

    viewer->setCurrentImage(viewer->getCurrentImage());
  }

  if (eventId == ImageViewer::currentImageId) {
    if (images.size()) {
      if (viewer->getCurrentImage() < 0 || viewer->getCurrentImage() >= (int)images.size())
        viewer->setCurrentImage(std::max(0, std::min(viewer->getCurrentImage(), (int)images.size() - 1)));
      else
        viewer->setCurrentSlice(viewer->getCurrentSlice());
    } else if (viewer->getCurrentImage()) {
      viewer->setCurrentImage(0);
    }
  }

  if (eventId == ImageViewer::currentSliceId) {
    if (images.size()) {
      if (viewer->getCurrentSlice() < 0 || viewer->getCurrentSlice() >= (int)images[viewer->getCurrentImage()]->getSize()[2])
        viewer->setCurrentSlice(std::max(0, std::min(viewer->getCurrentSlice(), (int)images[viewer->getCurrentImage()]->getSize()[2] - 1)));
      else
        updateView();
    } else if (viewer->getCurrentSlice()) {
      viewer->setCurrentSlice(0);
    }
  }

  if (eventId == ImageViewer::labelId || eventId == ImageViewer::minimumIntensityId || eventId == ImageViewer::maximumIntensityId) {
    updateView();
  }
}

ImageViewerDialog::ImageViewerDialog(ImageViewer* viewer) : QDialog(), widget(new ImageViewerWidget(viewer, this)) {
  QRect rect = geometry();
  rect.setWidth(widget->width());
  rect.setHeight(widget->height());
  setGeometry(rect);
  setWindowTitle((viewer->getLabel() + " (Image: 0/0; Slice: 0/0)").c_str());
  widget->setParent(this);
  widget->updateView();
}

void ImageViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

}

}

}
