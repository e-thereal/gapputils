#include "GridWidget.h"

#include "GridPointItem.h"
#include "GridLine.h"
#include "GridModel.h"

#include <QWheelEvent>
#include <QMouseEvent>

#include <vector>

namespace gapputils {

namespace cv {

GridWidget::GridWidget(GridModel* model, int width, int height, QWidget* parent) : QGraphicsView(parent),
    model(model), backgroundImage(0), viewScale(1.0)
{
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  scene->setSceneRect(0, 0, width, height);
  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));

  resumeFromModel();
}


GridWidget::~GridWidget(void)
{
}

void GridWidget::resumeFromModel() {
  const int rowCount = model->getRowCount();
  const int columnCount = model->getColumnCount();
  std::vector<GridPoint*>* points = model->getPoints();

  scene()->clear();

  std::vector<GridPointItem*> north(columnCount);
  for (unsigned i = 0; i < north.size(); ++i)
    north[i] = 0;
  GridPointItem* west = 0;

  for (int y = 0, i = 0; y < rowCount; ++y) {
    for (int x = 0; x < columnCount; ++x, ++i) {
      GridPoint* point = points->at(i);
      GridPointItem* item = new GridPointItem(point);
      scene()->addItem(item);

      if (west)
        scene()->addItem(new GridLine(west, item));
      if (north[x])
        scene()->addItem(new GridLine(north[x], item));

      west = item;
      north[x] = item;
    }
    west = 0;
  }
}

void GridWidget::updateSize(int width, int height) {
  scene()->setSceneRect(0, 0, width, height);
  renewGrid(model->getRowCount(), model->getColumnCount());
}

void GridWidget::renewGrid(int rowCount, int columnCount) {
  model->setRowCount(rowCount);
  model->setColumnCount(columnCount);
  scene()->clear();

  std::vector<GridPoint*>* points = model->getPoints();

  std::vector<GridPointItem*> north(columnCount);
  for (unsigned i = 0; i < north.size(); ++i)
    north[i] = 0;
  GridPointItem* west = 0;

  for (int y = 0, i = 0; y < rowCount; ++y) {
    for (int x = 0; x < columnCount; ++x, ++i) {
      GridPoint* point = points->at(i);
      point->setX(x * scene()->width() / (columnCount - 1));
      point->setY(y * scene()->height() / (rowCount - 1));
      GridPointItem* item = new GridPointItem(point);
      scene()->addItem(item);

      if (west)
        scene()->addItem(new GridLine(west, item));
      if (north[x])
        scene()->addItem(new GridLine(north[x], item));

      west = item;
      north[x] = item;
    }
    west = 0;
  }
}

void GridWidget::setBackgroundImage(QImage* image) {
  backgroundImage = image;
  setCacheMode(CacheNone);
  update();
  setCacheMode(CacheBackground);
}

void GridWidget::drawBackground(QPainter *painter, const QRectF &rect) {
  // Shadow
  QRectF& sceneRect = this->sceneRect();

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

void GridWidget::scaleView(qreal scaleFactor)
{
  viewScale *= scaleFactor;
  qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
  if (factor < 0.07 || factor > 100)
    return;

  scale(scaleFactor, scaleFactor);
}

void GridWidget::mousePressEvent(QMouseEvent* event) {
  

  if (event->button() == Qt::LeftButton)
    setDragMode(ScrollHandDrag);
  else if (event->button() == Qt::RightButton)
    setDragMode(RubberBandDrag);

  QGraphicsView::mousePressEvent(event);
}

void GridWidget::mouseReleaseEvent(QMouseEvent* event) {
  QGraphicsView::mouseReleaseEvent(event);
  setDragMode(NoDrag);
}

void GridWidget::wheelEvent(QWheelEvent *event)
{
  scaleView(pow((double)1.3, -event->delta() / 240.0));
}

}

}
