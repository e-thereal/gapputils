/*
 * Workbench.cpp
 *
 *  Created on: Mar 10, 2011
 *      Author: tombr
 */

#include "Workbench.h"

#include <QWheelEvent>
#include <cmath>
#include "ToolItem.h"
#include <QList>

using namespace std;

namespace gapputils {

Workbench::Workbench(QWidget *parent) : QGraphicsView(parent), selectedItem(0) {
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  scene->setSceneRect(-250, -250, 500, 500);
  setScene(scene);
  setCacheMode(CacheBackground);
  setRenderHint(QPainter::Antialiasing);
  setTransformationAnchor(AnchorUnderMouse);
  scale(qreal(1), qreal(1));
}

Workbench::~Workbench() {
}

void Workbench::addToolItem(ToolItem* item) {
  item->setWorkbench(this);
  scene()->addItem(item);
}

void Workbench::setSelectedItem(ToolItem* item) {
  selectedItem = item;

  Q_FOREACH (QGraphicsItem *item, scene()->items()) {
    item->update();
  }
  Q_EMIT itemSelected(item);
}

ToolItem* Workbench::getSelectedItem() {
  return selectedItem;
}

void Workbench::drawBackground(QPainter *painter, const QRectF &rect)
 {
     Q_UNUSED(rect);

     // Shadow
     QRectF sceneRect = this->sceneRect();
//     QRectF rightShadow(sceneRect.right(), sceneRect.top() + 5, 5, sceneRect.height());
//     QRectF bottomShadow(sceneRect.left() + 5, sceneRect.bottom(), sceneRect.width(), 5);
//     if (rightShadow.intersects(rect) || rightShadow.contains(rect))
//         painter->fillRect(rightShadow, Qt::darkGray);
//     if (bottomShadow.intersects(rect) || bottomShadow.contains(rect))
//         painter->fillRect(bottomShadow, Qt::darkGray);

     // Fill
     QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
     gradient.setColorAt(0, Qt::white);
     gradient.setColorAt(1, QColor(160, 160, 196));
     painter->fillRect(rect.intersect(sceneRect), gradient);
     painter->setBrush(Qt::NoBrush);
     painter->drawRect(sceneRect);

     // Text
     QRectF textRect(sceneRect.left() + 8, sceneRect.top() + 4,
                     sceneRect.width() - 16, sceneRect.height() - 8);
     QString message(tr("Application Host Workbench"));

     QFont font = painter->font();
     font.setBold(true);
     font.setPointSize(14);
     painter->setFont(font);
     painter->setPen(Qt::lightGray);
     painter->setOpacity(0.75);
     painter->drawText(textRect.translated(2, 2), Qt::AlignBottom | Qt::AlignRight, message);
     painter->setPen(Qt::black);
     painter->drawText(textRect, Qt::AlignBottom | Qt::AlignRight, message);
 }

void Workbench::wheelEvent(QWheelEvent *event)
 {
     scaleView(pow((double)2, -event->delta() / 240.0));
 }

void Workbench::scaleView(qreal scaleFactor)
 {
   qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
   if (factor < 0.07 || factor > 100)
       return;

   scale(scaleFactor, scaleFactor);
 }

}
