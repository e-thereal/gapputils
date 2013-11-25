/*
 * TensorViewerDialog.cpp
 *
 *  Created on: Nov 22, 2013
 *      Author: tombr
 */

#include "TensorViewerDialog.h"

#include <QWheelEvent>
#include <QMouseEvent>
#include <qfiledialog.h>
#include <qpen.h>

#include "TensorViewer.h"

#include <capputils/EventHandler.h>

#include <sstream>
#include <cmath>

#include "math3d.h"

using namespace gapputils;

namespace gml {

namespace imaging {

namespace ui {

TensorViewerWidget::TensorViewerWidget(TensorViewer* viewer, TensorViewerDialog* dialog)
 : QGraphicsView(), viewer(viewer), dialog(dialog), viewScale(1.0)
{
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  if (viewer->getTensor()) {
    tensor_t& tensor = *viewer->getTensor();
    scene->setSceneRect(0, 0, tensor.size()[0], tensor.size()[1]);
  } else {
    scene->setSceneRect(0, 0, 128, 128);
  }

  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));

  viewer->Changed.connect(capputils::EventHandler<TensorViewerWidget>(this, &TensorViewerWidget::changedHandler));

  tensors.clear();
  if (viewer->getTensor())
    tensors.push_back(viewer->getTensor());
  if (viewer->getTensors()) {
    for (size_t i = 0; i < viewer->getTensors()->size(); ++i)
      tensors.push_back(viewer->getTensors()->at(i));
  }

  viewer->setCurrentTensor(viewer->getCurrentTensor());
}

#define F_TO_INT(value) std::min(255, std::max(0, (int)((value) * 256)))

void TensorViewerWidget::updateBackground() {
  if (viewer->getBackground()) {
    image_t& image = *viewer->getBackground();
//    scene()->setSceneRect(0, 0, image.getSize()[0], image.getSize()[1]);

    const int width = image.getSize()[0];
    const int height = image.getSize()[1];
  //    const int depth = image.getSize()[2];

    const int count = width * height;
    qimage = boost::make_shared<QImage>(width, height, QImage::Format_ARGB32);

    int slicePos = image.getSize()[2] / 2;

    if (tensors.size() && tensors[viewer->getCurrentTensor()]->size()[2]) {
      tensor_t& tensor = *tensors[viewer->getCurrentTensor()];
      slicePos = std::min(image.getSize()[2] - 1, viewer->getCurrentSlice() * image.getSize()[2] / tensor.size()[2]);
    }

    float* buffer = image.getData();
    for (int i = 0, y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {

        int c = F_TO_INT(buffer[i + slicePos * count] / 1024.0);
        qimage->setPixel(x, y, QColor(c, c, c).rgb());
      }
    }
    setCacheMode(CacheNone);
    update();
    setCacheMode(CacheBackground);
  }
}

void TensorViewerWidget::updateView() {
  using namespace tbblas;

  std::stringstream title;
  title << viewer->getLabel() << " (Tensor: ";
  if (tensors.size())
    title << viewer->getCurrentTensor() + 1 << "/" << tensors.size();
  else
    title << "0/0";
  title << ";" << "Slice: ";
  if (tensors.size() && tensors[viewer->getCurrentTensor()]->size()[2])
    title << viewer->getCurrentSlice() + 1 << "/" << tensors[viewer->getCurrentTensor()]->size()[2];
  else
    title << "0/0";
  title << ")";
  dialog->setWindowTitle(title.str().c_str());

  if (tensors.size() && tensors[viewer->getCurrentTensor()]->size()[2]) {
    tensor_t& tensor = *tensors[viewer->getCurrentTensor()];
    QGraphicsScene* scene = new QGraphicsScene();
    scene->setSceneRect(0, 0, tensor.size()[0], tensor.size()[1]);
    scene->addRect(0, 0, tensor.size()[0], tensor.size()[1]);

    const int width = tensor.size()[0];
    const int height = tensor.size()[1];

    float w = 0.05, h = 0.05;

    for (int i = 0, y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {
        float dx = (tensor[seq(x,y,viewer->getCurrentSlice(),0)] - viewer->getMinimumLength()) / (viewer->getMaximumLength() - viewer->getMinimumLength()) * viewer->getVisibleLength();
        float dy = (tensor[seq(x,y,viewer->getCurrentSlice(),1)] - viewer->getMinimumLength()) / (viewer->getMaximumLength() - viewer->getMinimumLength()) * viewer->getVisibleLength();
        scene->addLine(x + 0.5, y + 0.5, x + dx + 0.5, y + dy + 0.5, QPen(Qt::red));
        scene->addEllipse(x - w + 0.5, y - h + 0.5, 2 * w, 2 * h, QPen(Qt::red), QBrush(Qt::red));
      }
    }
    QGraphicsScene* oldScene = this->scene();
    setScene(scene);
    delete oldScene;
  }
  update();
}

void TensorViewerWidget::drawBackground(QPainter *painter, const QRectF &rect) {
//  painter->fillRect(rect, Qt::black);
  if (qimage) {
    painter->drawImage(scene()->sceneRect(), *qimage);
  }

//  QGraphicsView::drawBackground(painter, rect);
}

void TensorViewerWidget::scaleView(qreal scaleFactor)
{
  viewScale *= scaleFactor;
  qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
  if (factor < 0.07 || factor > 100)
    return;

  scale(scaleFactor, scaleFactor);
}

void TensorViewerWidget::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton)
    setDragMode(ScrollHandDrag);
  else if (event->button() == Qt::RightButton)
    setDragMode(RubberBandDrag);

  dragStart = mapToScene(event->pos());

  QGraphicsView::mousePressEvent(event);
}

void TensorViewerWidget::mouseReleaseEvent(QMouseEvent* event) {
  using namespace tbblas;

  if (event->button() == Qt::RightButton) {
    QPointF dragEnd = mapToScene(event->pos());
    if (tensors.size()) {
      tensor_t& tensor = *tensors[viewer->getCurrentTensor()];

      const int width = tensor.size()[0], height = tensor.size()[1], depth = tensor.size()[2], count = width * height;

      const int rx = std::max(0, std::min(width - 1, (int)std::min(dragStart.x(), dragEnd.x())));
      const int ry = std::max(0, std::min(height - 1, (int)std::min(dragStart.y(), dragEnd.y())));
      const int rwidth = std::max(0, std::min(width - rx, (int)std::max(dragStart.x(), dragEnd.x()) - rx));
      const int rheight = std::max(0, std::min(height - ry, (int)std::max(dragStart.y(), dragEnd.y()) - ry));

      float minimum, maximum, length, dx, dy;

      if (event->modifiers() == Qt::ControlModifier) {
        minimum = viewer->getMinimumLength();
        maximum = viewer->getMaximumLength();
      } else {
        dx = tensor[seq(0,0,viewer->getCurrentSlice(),0)];
        dy = tensor[seq(0,0,viewer->getCurrentSlice(),1)];
        minimum = maximum = length = sqrt(dx * dx + dy * dy);
      }

      for (int y = ry; y < ry + rheight; ++y) {
        for (int x = rx; x < rx + rwidth; ++x) {
          dx = tensor[seq(x,y,viewer->getCurrentSlice(),0)];
          dy = tensor[seq(x,y,viewer->getCurrentSlice(),1)];
          length = sqrt(dx * dx + dy * dy);

          minimum = std::min(minimum, length);
          maximum = std::max(maximum, length);
        }
      }

      // TODO: Make the update somewhat atomic
      viewer->setMinimumLength(minimum);
      viewer->setMaximumLength(maximum);
    }
  }

  QGraphicsView::mouseReleaseEvent(event);
  setDragMode(NoDrag);
}

void TensorViewerWidget::wheelEvent(QWheelEvent *event) {
  scaleView(pow((double)1.3, event->delta() / 240.0));
}

void TensorViewerWidget::keyPressEvent(QKeyEvent *event) {
//  if (event->key() == Qt::Key_S && event->modifiers() == Qt::CTRL) {
//    if (qimage) {
//      QString filename = QFileDialog::getSaveFileName(this, "Save current image as ...");
//      if (filename.size()) {
//        qimage->save(filename);
//      }
//    }
//    return;
//  }

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
    viewer->setCurrentTensor(viewer->getCurrentTensor() - 1);
    return;

  case Qt::Key_Right:
  case Qt::Key_F:
    viewer->setCurrentTensor(viewer->getCurrentTensor() + 1);
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

qreal TensorViewerWidget::getViewScale() {
  return viewScale;
}


void TensorViewerWidget::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == TensorViewer::tensorId || eventId == TensorViewer::tensorsId) {
    tensors.clear();
    if (viewer->getTensor())
      tensors.push_back(viewer->getTensor());
    if (viewer->getTensors()) {
      for (size_t i = 0; i < viewer->getTensors()->size(); ++i)
        tensors.push_back(viewer->getTensors()->at(i));
    }

    viewer->setCurrentTensor(viewer->getCurrentTensor());
  }

  if (eventId == TensorViewer::currentTensorId) {
    if (tensors.size()) {
      if (viewer->getCurrentTensor() < 0 || viewer->getCurrentTensor() >= (int)tensors.size())
        viewer->setCurrentTensor(std::max(0, std::min(viewer->getCurrentTensor(), (int)tensors.size() - 1)));
      else
        viewer->setCurrentSlice(viewer->getCurrentSlice());
    } else if (viewer->getCurrentTensor()) {
      viewer->setCurrentTensor(0);
    }
  }

  if (eventId == TensorViewer::currentSliceId) {
    if (tensors.size() && (int)tensors[viewer->getCurrentTensor()]->size()[2]) {
      if (viewer->getCurrentSlice() < 0 || viewer->getCurrentSlice() >= (int)tensors[viewer->getCurrentTensor()]->size()[2]) {
        viewer->setCurrentSlice(std::max(0, std::min(viewer->getCurrentSlice(), (int)tensors[viewer->getCurrentTensor()]->size()[2] - 1)));
      } else {
        updateBackground();
        updateView();
      }
    } else if (viewer->getCurrentSlice()) {
      viewer->setCurrentSlice(0);
    }
  }

  if (eventId == TensorViewer::labelId || eventId == TensorViewer::minimumLengthId || eventId == TensorViewer::maximumLengthId) {
    updateView();
  }

  if (eventId == TensorViewer::backgroundId)
    updateBackground();
}

TensorViewerDialog::TensorViewerDialog(TensorViewer* viewer) : QDialog(),
    widget(new TensorViewerWidget(viewer, this)), helpLabel(new QLabel())
{
  QRect rect = geometry();
  rect.setWidth(widget->width());
  rect.setHeight(widget->height());
  setGeometry(rect);
  setWindowTitle((viewer->getLabel() + " (Tensor: 0/0; Slice: 0/0)").c_str());
  widget->setParent(this);
  widget->updateView();
  widget->updateBackground();
  helpLabel->setParent(this);
  helpLabel->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
  helpLabel->setStyleSheet("color: rgb(255, 255, 255); background-color: rgba(0, 0, 0, 210);");
  helpLabel->setText("<h3>Hotkeys</h3>\n"
      "<b>Space, H, F1:</b> Toggle help<br>\n"
      "<b>S, arrow left:</b> View previous tensor<br>\n"
      "<b>F, arrow right:</b> View next tensor<br>\n"
      "<b>E, arrow up:</b> Show next slice<br>\n"
      "<b>D, arrow down:</b> Show previous slice<br>\n"
      "<b>W:</b> Resize window to match the tensor size<br>\n"
      "<b>R:</b> Fit the tensor into the window<br>\n"
      "<b>Q, Esc:</b> Close the viewer<br>\n"
//      "<b>Ctrl+S:</b> Save the current image<br>\n"
      "<h3>Mouse Actions</h3>\n"
      "<b>Left drag:</b> Panning<br>\n"
      "<b>Right drag:</b> Maximize contrast in the selected region<br>\n"
      "<b>Ctrl + right drag:</b> Add new region to intensity window<br>\n"
      "<b>Mouse wheel:</b> Adjust zoom<br>\n");
  helpLabel->setGeometry(0, 0, width(), height());
}

void TensorViewerDialog::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Space || event->key() == Qt::Key_H || event->key() == Qt::Key_F1) {
//    if (!helpLabel->isVisible()) {
      helpLabel->setVisible(!helpLabel->isVisible());
//    }
    return;
  }
  QDialog::keyPressEvent(event);
}

void TensorViewerDialog::keyReleaseEvent(QKeyEvent *event) {
  // even if you don't release the key, after a while multiple key press and key release events are issued.
//  if (event->key() == Qt::Key_Space) {
//    std::cout << "Key released." << std::endl;
//    helpLabel->setVisible(false);
//    return;
//  }
  QDialog::keyReleaseEvent(event);
}

void TensorViewerDialog::resizeEvent(QResizeEvent* resizeEvent) {
  widget->setGeometry(0, 0, width(), height());
  helpLabel->setGeometry(0, 0, width(), height());

  QDialog::resizeEvent(resizeEvent);
}

}

}

}
