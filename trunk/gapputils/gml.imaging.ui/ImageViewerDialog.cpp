/*
 * ImageViewerDialog.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewerDialog.h"
#include <QWheelEvent>
#include <QMouseEvent>
#include <qfiledialog.h>

#include "ImageViewer.h"

#include <capputils/EventHandler.h>

#include <sstream>
#include <cmath>

#include "math3d.h"

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

  //if (viewer->getMode() == ViewMode::Wobble) {
  //  timer = boost::make_shared<QTimer>(this);
  //  connect(timer.get(), SIGNAL(timeout()), this, SLOT(updateView()));
  //  timer->start(viewer->getWobbleDelay());
  //}

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

#define F_TO_INT(value) std::min(255, std::max(0, (int)((value) * 256)))

void getHeatMap1Color(float value, float *red, float *green, float *blue) {
  const int NUM_COLORS = 4;
  static float color[NUM_COLORS][3] = { {0,0,1}, {0,1,0}, {1,1,0}, {1,0,0} };
    // a static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each

  int idx1;        // |-- our desired color will be between these two indexes in "color"
  int idx2;        // |
  float fractBetween = 0;  // fraction between "idx1" and "idx2" where our value is

  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // will multiply value by 3
    idx1  = floor(value);                  // our desired color will be after this index
    idx2  = idx1+1;                        // ... and before this index (inclusive)
    fractBetween = value - float(idx1);    // distance between the two indexes (0-1)
  }

  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

void getHeatMap2Color(float value, float *red, float *green, float *blue) {
  const int NUM_COLORS = 6;
  static float color[NUM_COLORS][3] = {{0,0,0}, {0,0,0.5}, {0.5,0,0.5}, {1,0,0}, {1,1,0}, {1,1,1}};
    // a static array of 6 colors:  (black,      blue,   violet,  red,  yellow,     white) using {r,g,b} for each

  int idx1;        // |-- our desired color will be between these two indexes in "color"
  int idx2;        // |
  float fractBetween = 0;  // fraction between "idx1" and "idx2" where our value is

  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // will multiply value by 3
    idx1  = floor(value);                  // our desired color will be after this index
    idx2  = idx1+1;                        // ... and before this index (inclusive)
    fractBetween = value - float(idx1);    // distance between the two indexes (0-1)
  }

  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

void ImageViewerWidget::updateView() {
  // D65 reference white
  const float Xr = 0.95047f, Yr = 1.00000f, Zr = 1.08883f;
  const float eps = 0.008856f, kappa = 903.3f;

  int channelsPerImage = 1;
  switch (viewer->getMode()) {
  case ViewMode::XYZ:
  case ViewMode::xyY:
  case ViewMode::sRGB:
  case ViewMode::CIELAB:
    channelsPerImage = 3;
    break;
  }

  std::stringstream title;
  title << viewer->getLabel() << " (Image: ";
  if (images.size())
    title << viewer->getCurrentImage() + 1 << "/" << images.size();
  else
    title << "0/0";
  title << ";" << "Slice: ";
  if (images.size() && images[viewer->getCurrentImage()]->getSize()[2] / channelsPerImage)
    title << viewer->getCurrentSlice() + 1 << "/" << images[viewer->getCurrentImage()]->getSize()[2] / channelsPerImage;
  else
    title << "0/0";
  title << ")";
  dialog->setWindowTitle(title.str().c_str());

  if (images.size() && images[viewer->getCurrentImage()]->getSize()[2] / channelsPerImage) {
    image_t& image = *images[viewer->getCurrentImage()];
    scene()->setSceneRect(0, 0, image.getSize()[0], image.getSize()[1]);

    const int width = image.getSize()[0];
    const int height = image.getSize()[1];
//    const int depth = image.getSize()[2];

    const int count = width * height;
    qimage = boost::make_shared<QImage>(width, height, QImage::Format_ARGB32);

    float* buffer = image.getData();

    switch (viewer->getMode()) {
    case ViewMode::Greyscale:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int c = F_TO_INT((buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          qimage->setPixel(x, y, QColor(c, c, c).rgb());
        }
      }
      break;

    case ViewMode::RedBlueMap:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int r = F_TO_INT((buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          int b = F_TO_INT(1.0 - (buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          qimage->setPixel(x, y, QColor(r, 0, b).rgb());
        }
      }
      break;

    case ViewMode::HeatMap1:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          float r, g, b;
          getHeatMap1Color((buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()), &r, &g, &b);
          qimage->setPixel(x, y, QColor(F_TO_INT(r), F_TO_INT(g), F_TO_INT(b)).rgb());
        }
      }
      break;

    case ViewMode::HeatMap2:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          float r, g, b;
          getHeatMap2Color((buffer[i + viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()), &r, &g, &b);
          qimage->setPixel(x, y, QColor(F_TO_INT(r), F_TO_INT(g), F_TO_INT(b)).rgb());
        }
      }
      break;

    case ViewMode::sRGB:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          int R = F_TO_INT((buffer[i + 3 * viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          int G = F_TO_INT((buffer[i + (3 * viewer->getCurrentSlice() + 1) * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          int B = F_TO_INT((buffer[i + (3 * viewer->getCurrentSlice() + 2) * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()));
          qimage->setPixel(x, y, QColor(R, G, B).rgb());
        }
      }
      break;

    case ViewMode::XYZ:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          const float X = (buffer[i + 3 * viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity());
          const float Y = (buffer[i + (3 * viewer->getCurrentSlice() + 1) * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity());
          const float Z = (buffer[i + (3 * viewer->getCurrentSlice() + 2) * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity());

          gml::fmatrix4 xyz2rgb = gml::make_fmatrix4(3.2404542f, -1.5371385f, -0.4985314f, 0,
                                                    -0.9692660f,  1.8760108f,  0.0415560f, 0,
                                                     0.055644f, -0.2040259f,  1.0572252f, 0,
                                                     0, 0, 0, 1);
          gml::float4 xyz = make_float4(X, Y, Z, 1);
          gml::float4 rgb = xyz2rgb * xyz;

          const float r = gml::get_x(rgb);
          const float g = gml::get_y(rgb);
          const float b = gml::get_z(rgb);
          const int R = F_TO_INT((r <= 0.00313088 ? 12.92 * r : 1.055 * powf(r, 1.f / 2.4f) - 0.055));
          const int G = F_TO_INT((g <= 0.00313088 ? 12.92 * g : 1.055 * powf(g, 1.f / 2.4f) - 0.055));
          const int B = F_TO_INT((b <= 0.00313088 ? 12.92 * b : 1.055 * powf(b, 1.f / 2.4f) - 0.055));

          qimage->setPixel(x, y, QColor(R, G, B).rgb());
        }
      }
      break;

    case ViewMode::xyY:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          const float cx = buffer[i + 3 * viewer->getCurrentSlice() * count];
          const float cy = buffer[i + (3 * viewer->getCurrentSlice() + 1) * count];
          const float Y = (buffer[i + (3 * viewer->getCurrentSlice() + 2) * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity());

          const float X = Y / cy * cx;
          const float Z = Y / cy * (1 - cx - cy);

          gml::fmatrix4 xyz2rgb = gml::make_fmatrix4(3.2404542f, -1.5371385f, -0.4985314f, 0,
                                                    -0.9692660f,  1.8760108f,  0.0415560f, 0,
                                                     0.055644f, -0.2040259f,  1.0572252f, 0,
                                                     0, 0, 0, 1);
          gml::float4 xyz = make_float4(X, Y, Z, 1);
          gml::float4 rgb = xyz2rgb * xyz;

          const float r = gml::get_x(rgb);
          const float g = gml::get_y(rgb);
          const float b = gml::get_z(rgb);
          int R = F_TO_INT((r <= 0.00313088 ? 12.92 * r : 1.055 * powf(r, 1.f / 2.4f) - 0.055));
          int G = F_TO_INT((g <= 0.00313088 ? 12.92 * g : 1.055 * powf(g, 1.f / 2.4f) - 0.055));
          int B = F_TO_INT((b <= 0.00313088 ? 12.92 * b : 1.055 * powf(b, 1.f / 2.4f) - 0.055));

          qimage->setPixel(x, y, QColor(R, G, B).rgb());
        }
      }
      break;

    case ViewMode::CIELAB:
      for (int i = 0, y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          const float L = (buffer[i + 3 * viewer->getCurrentSlice() * count] - viewer->getMinimumIntensity()) / (viewer->getMaximumIntensity() - viewer->getMinimumIntensity()) * 100.0;
          const float a = buffer[i + (3 * viewer->getCurrentSlice() + 1) * count] * 100.0;
          const float b = buffer[i + (3 * viewer->getCurrentSlice() + 2) * count] * 100.0;

          const float fy = (L + 16.f) / 116.f;
          const float fx = a / 500.f + fy;
          const float fz = fy - b / 200.f;

          float xr = fx * fx * fx;
          if (xr <= eps)
            xr = (116 * fx - 16) / kappa;

          float yr = 0.f;
          if (L > kappa * eps) {
            yr = (L + 16.f) / 116.f;
            yr = yr * yr * yr;
          } else {
            yr = L / kappa;
          }

          float zr = fz * fz * fz;
          if (zr <= eps)
            zr = (116 * fz - 16) / kappa;

          const float X = xr * Xr;
          const float Y = yr * Yr;
          const float Z = zr * Zr;

          gml::fmatrix4 xyz2rgb = gml::make_fmatrix4(3.2404542f, -1.5371385f, -0.4985314f, 0,
                                                    -0.9692660f,  1.8760108f,  0.0415560f, 0,
                                                     0.055644f, -0.2040259f,  1.0572252f, 0,
                                                     0, 0, 0, 1);
          gml::float4 xyz = make_float4(X, Y, Z, 1);
          gml::float4 rgb = xyz2rgb * xyz;

          
          const float r2 = gml::get_x(rgb);
          const float g2 = gml::get_y(rgb);
          const float b2 = gml::get_z(rgb);
          int R = F_TO_INT((r2 <= 0.00313088 ? 12.92 * r2 : 1.055 * powf(r2, 1.f / 2.4f) - 0.055));
          int G = F_TO_INT((g2 <= 0.00313088 ? 12.92 * g2 : 1.055 * powf(g2, 1.f / 2.4f) - 0.055));
          int B = F_TO_INT((b2 <= 0.00313088 ? 12.92 * b2 : 1.055 * powf(b2, 1.f / 2.4f) - 0.055));

          qimage->setPixel(x, y, QColor(R, G, B).rgb());
          
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
  painter->fillRect(rect, Qt::black);
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

      const int width = image.getSize()[0], height = image.getSize()[1], depth = image.getSize()[2], count = width * height;

      const int rx = std::max(0, std::min(width - 1, (int)std::min(dragStart.x(), dragEnd.x())));
      const int ry = std::max(0, std::min(height - 1, (int)std::min(dragStart.y(), dragEnd.y())));
      const int rwidth = std::max(0, std::min(width - rx, (int)std::max(dragStart.x(), dragEnd.x()) - rx));
      const int rheight = std::max(0, std::min(height - ry, (int)std::max(dragStart.y(), dragEnd.y()) - ry));

      float* buffer = image.getData();

      float minimum, maximum;

      switch (viewer->getMode()) {

      case ViewMode::Greyscale:
      case ViewMode::RedBlueMap:
      case ViewMode::HeatMap1:
      case ViewMode::HeatMap2:
        if (depth < 1)
          break;

        if (event->modifiers() == Qt::ControlModifier) {
          minimum = viewer->getMinimumIntensity();
          maximum = viewer->getMaximumIntensity();
        } else {
          minimum = maximum = buffer[viewer->getCurrentSlice() * count + ry * width + rx];
        }

        for (int y = ry; y < ry + rheight; ++y) {
          for (int x = rx; x < rx + rwidth; ++x) {
            minimum = std::min(minimum, buffer[viewer->getCurrentSlice() * count + y * width + x]);
            maximum = std::max(maximum, buffer[viewer->getCurrentSlice() * count + y * width + x]);
          }
        }
        break;

      case ViewMode::sRGB:
      case ViewMode::XYZ:
        if (depth < 3)
          break;
        if (event->modifiers() == Qt::ControlModifier) {
          minimum = viewer->getMinimumIntensity();
          maximum = viewer->getMaximumIntensity();
        } else {
          minimum = maximum = buffer[(3 * viewer->getCurrentSlice()) * count + ry * width + rx];
        }

        for (int y = ry; y < ry + rheight; ++y) {
          for (int x = rx; x < rx + rwidth; ++x) {
            minimum = std::min(minimum, buffer[(3 * viewer->getCurrentSlice()) * count + y * width + x]);
            minimum = std::min(minimum, buffer[(3 * viewer->getCurrentSlice() + 1) * count + y * width + x]);
            minimum = std::min(minimum, buffer[(3 * viewer->getCurrentSlice() + 2) * count + y * width + x]);
            
            maximum = std::max(maximum, buffer[(3 * viewer->getCurrentSlice()) * count + y * width + x]);
            maximum = std::max(maximum, buffer[(3 * viewer->getCurrentSlice() + 1) * count + y * width + x]);
            maximum = std::max(maximum, buffer[(3 * viewer->getCurrentSlice() + 2) * count + y * width + x]);
          }
        }
        break;

      case ViewMode::xyY:
        if (depth < 3)
          break;
        if (event->modifiers() == Qt::ControlModifier) {
          minimum = viewer->getMinimumIntensity();
          maximum = viewer->getMaximumIntensity();
        } else {
          minimum = maximum = buffer[(3 * viewer->getCurrentSlice() + 2) * count + ry * width + rx];
        }

        for (int y = ry; y < ry + rheight; ++y) {
          for (int x = rx; x < rx + rwidth; ++x) {
            minimum = std::min(minimum, buffer[(3 * viewer->getCurrentSlice() + 2) * count + y * width + x]);
            maximum = std::max(maximum, buffer[(3 * viewer->getCurrentSlice() + 2) * count + y * width + x]);
          }
        }
        break;

      case ViewMode::CIELAB:
        if (depth < 3)
          break;
        if (event->modifiers() == Qt::ControlModifier) {
          minimum = viewer->getMinimumIntensity();
          maximum = viewer->getMaximumIntensity();
        } else {
          minimum = maximum = buffer[3 * viewer->getCurrentSlice() * count + ry * width + rx];
        }

        for (int y = ry; y < ry + rheight; ++y) {
          for (int x = rx; x < rx + rwidth; ++x) {
            minimum = std::min(minimum, buffer[3 * viewer->getCurrentSlice() * count + y * width + x]);
            maximum = std::max(maximum, buffer[3 * viewer->getCurrentSlice() * count + y * width + x]);
          }
        }
        break;
      }

      float contrastMargin = (maximum - minimum) * (1.0 - viewer->getContrast()) / 2;

      // TODO: Makes the update somewhat atomic
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
  if (event->key() == Qt::Key_S && event->modifiers() == Qt::CTRL) {
    if (qimage) {
      QString filename = QFileDialog::getSaveFileName(this, "Save current image as ...");
      if (filename.size()) {
        qimage->save(filename);
      }
    }
    return;
  }

  switch (event->key()) {
  case Qt::Key_Q:
  case Qt::Key_Escape:
    dialog->close();
    return;

  case Qt::Key_W:
    {
      QRect rect = dialog->geometry();
      rect.setWidth(scene()->sceneRect().width() * viewScale + 5);
      rect.setHeight(scene()->sceneRect().height() * viewScale + 5);
      dialog->setGeometry(rect);
    }
    return;

  case Qt::Key_R:
    scaleView(std::min((qreal)(dialog->geometry().width() - 5.0) / (qreal)scene()->sceneRect().width(),
        (qreal)(dialog->geometry().height() - 5.0) / (qreal)scene()->sceneRect().height()) / viewScale);
    return;

  case Qt::Key_Left:
  case Qt::Key_S:
    viewer->setCurrentImage(viewer->getCurrentImage() - 1);
    return;

  case Qt::Key_Right:
  case Qt::Key_F:
    viewer->setCurrentImage(viewer->getCurrentImage() + 1);
    return;

  case Qt::Key_Up:
  case Qt::Key_E:
    viewer->setCurrentSlice(viewer->getCurrentSlice() + 1);
    return;

  case Qt::Key_Down:
  case Qt::Key_D:
    viewer->setCurrentSlice(viewer->getCurrentSlice() - 1);
    return;
  }
  QGraphicsView::keyPressEvent(event);
}

qreal ImageViewerWidget::getViewScale() {
  return viewScale;
}


void ImageViewerWidget::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {

  if (eventId == ImageViewer::modeId) {
    //if (viewer->getMode() == ViewMode::Wobble) {
    //  timer = boost::make_shared<QTimer>(this);
    //  connect(timer.get(), SIGNAL(timeout()), this, SLOT(updateView()));
    //  timer->start(viewer->getWobbleDelay());
    //} else {
    //  timer = boost::shared_ptr<QTimer>();
    //}
    viewer->setCurrentImage(viewer->getCurrentImage());
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
    int channelCount = 1;
    switch (viewer->getMode()) {
      case ViewMode::XYZ:
      case ViewMode::xyY:
      case ViewMode::sRGB:
      case ViewMode::CIELAB:
        channelCount = 3;
        break;
    }
    if (images.size() && (int)images[viewer->getCurrentImage()]->getSize()[2] / channelCount) {
      if (viewer->getCurrentSlice() < 0 || viewer->getCurrentSlice() >= (int)images[viewer->getCurrentImage()]->getSize()[2] / channelCount)
        viewer->setCurrentSlice(std::max(0, std::min(viewer->getCurrentSlice(), (int)images[viewer->getCurrentImage()]->getSize()[2] / channelCount - 1)));
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

ImageViewerDialog::ImageViewerDialog(ImageViewer* viewer) : QDialog(),
    widget(new ImageViewerWidget(viewer, this)), helpLabel(new QLabel())
{
  QRect rect = geometry();
  rect.setWidth(widget->width());
  rect.setHeight(widget->height());
  setGeometry(rect);
  setWindowTitle((viewer->getLabel() + " (Image: 0/0; Slice: 0/0)").c_str());
  widget->setParent(this);
  widget->updateView();
  helpLabel->setParent(this);
  helpLabel->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
  helpLabel->setStyleSheet("color: rgb(255, 255, 255); background-color: rgba(0, 0, 0, 210);");
  helpLabel->setText("<h3>Hotkeys</h3>\n"
      "<b>Space, H, F1:</b> Toggle help<br>\n"
      "<b>S, arrow left:</b> View previous image<br>\n"
      "<b>F, arrow right:</b> View next image<br>\n"
      "<b>E, arrow up:</b> Show next slice<br>\n"
      "<b>D, arrow down:</b> Show previous slice<br>\n"
      "<b>W:</b> Resize window to match the image size<br>\n"
      "<b>R:</b> Fit the image into the window<br>\n"
      "<b>Q, Esc:</b> Close the viewer<br>\n"
      "<b>Ctrl+S:</b> Save the current image<br>\n"
      "<h3>Mouse Actions</h3>\n"
      "<b>Left drag:</b> Panning<br>\n"
      "<b>Right drag:</b> Maximize contrast in the selected region<br>\n"
      "<b>Ctrl + right drag:</b> Add new region to intensity window<br>\n"
      "<b>Mouse wheel:</b> Adjust zoom<br>\n");
  helpLabel->setGeometry(0, 0, width(), height());
}

void ImageViewerDialog::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Space || event->key() == Qt::Key_H || event->key() == Qt::Key_F1) {
//    if (!helpLabel->isVisible()) {
      helpLabel->setVisible(!helpLabel->isVisible());
//    }
    return;
  }
  QDialog::keyPressEvent(event);
}

void ImageViewerDialog::keyReleaseEvent(QKeyEvent *event) {
  // even if you don't release the key, after a while multiple key press and key release events are issued.
//  if (event->key() == Qt::Key_Space) {
//    std::cout << "Key released." << std::endl;
//    helpLabel->setVisible(false);
//    return;
//  }
  QDialog::keyReleaseEvent(event);
}

void ImageViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());
  helpLabel->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

}

}

}
