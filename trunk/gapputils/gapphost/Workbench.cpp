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
#include <qmessagebox.h>
#include <iostream>
#include "CableItem.h"

using namespace std;

namespace gapputils {

Workbench::Workbench(QWidget *parent) : QGraphicsView(parent), selectedItem(0), currentCable(0) {
  QGraphicsScene *scene = new QGraphicsScene(this);
  scene->setItemIndexMethod(QGraphicsScene::NoIndex);
  scene->setSceneRect(-250, -250, 500, 500);
  //scene->addItem(new CablePlug());
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

void Workbench::addCableItem(CableItem* cable) {
  scene()->addItem(cable);
}

void Workbench::removeCableItem(CableItem* cable) {
  // TODO: Workaround. When loading workflows, you get an error here. Cable seems to be already removed
  if (cable->scene() == scene()) {
    scene()->removeItem(cable);
    Q_EMIT cableDeleted(cable);
    delete cable;
  }
}

void Workbench::setSelectedItem(ToolItem* item) {
  selectedItem = item;

  Q_FOREACH (QGraphicsItem *item, scene()->items()) {
    item->update();
  }
  Q_EMIT itemSelected(item);
}

ToolItem* Workbench::getSelectedItem() const {
  return selectedItem;
}

CableItem* Workbench::getCurrentCable() const {
  return currentCable;
}

void Workbench::notifyItemChange(ToolItem* item) {
  Q_EMIT itemChanged(item);
}

void Workbench::mousePressEvent(QMouseEvent* event) {
  Q_FOREACH(QGraphicsItem* item, scene()->items()) {
    ToolItem* tool = dynamic_cast<ToolItem*>(item);
    if (tool) {
      QPointF mousePos = mapToScene(event->pos());
      int ex = mousePos.x();
      int ey = mousePos.y();
      int tx = item->x();
      int ty = item->y();
      ToolConnection* connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Output);
      if (connection) {
        if (connection->cable) {
          currentCable = connection->cable;
          currentCable->disconnectInput();
          currentCable->setDragPoint(mapToScene(event->pos()));
          Q_EMIT cableDeleted(currentCable);
        } else {
          currentCable = new CableItem(this, connection);
          currentCable->setDragPoint(mapToScene(event->pos()));
          scene()->addItem(currentCable);
        }
        Q_FOREACH (QGraphicsItem *item, scene()->items()) {
          item->update();
        }
      }
      connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Input);
      if (connection) {
        if (connection->cable) {
          currentCable = connection->cable;
          currentCable->disconnectOutput();
          currentCable->setDragPoint(mapToScene(event->pos()));
          Q_EMIT cableDeleted(currentCable);
        } else {
          currentCable = new CableItem(this, 0, connection);
          currentCable->setDragPoint(mapToScene(event->pos()));
          scene()->addItem(currentCable);
        }
        Q_FOREACH (QGraphicsItem *item, scene()->items()) {
          item->update();
        }
      }
    }
  }
  QGraphicsView::mousePressEvent(event);
}

void Workbench::mouseReleaseEvent(QMouseEvent* event) {
  if (currentCable) {
    bool foundConnection = false;
    Q_FOREACH(QGraphicsItem* item, scene()->items()) {
      ToolItem* tool = dynamic_cast<ToolItem*>(item);
      if (tool) {
        QPointF mousePos = mapToScene(event->pos());
        int ex = mousePos.x();
        int ey = mousePos.y();
        int tx = item->x();
        int ty = item->y();
        if (currentCable->needOutput()) {
          ToolConnection* connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Input);
          if (connection && connection->property->getType() == currentCable->getInput()->property->getType()) {
            foundConnection = true;
            currentCable->setOutput(connection);
            currentCable->endDrag();
            Q_EMIT cableCreated(currentCable);
            break;
          }
        } else if (currentCable->needInput()) {
          ToolConnection* connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Output);
          if (connection && connection->property->getType() == currentCable->getOutput()->property->getType()) {
            foundConnection = true;
            currentCable->setInput(connection);
            currentCable->endDrag();
            Q_EMIT cableCreated(currentCable);
            break;
          }
        }
      }
    }
    if (!foundConnection) {
      scene()->removeItem(currentCable);
      delete currentCable;
    }
    currentCable = 0;
    Q_FOREACH (QGraphicsItem *item, scene()->items()) {
      item->update();
    }
  }
  QGraphicsView::mouseReleaseEvent(event);
}

void Workbench::keyPressEvent(QKeyEvent *event)
{
  switch (event->key()) {
  case Qt::Key_Delete:
    if (selectedItem && selectedItem->isDeletable()) {
      scene()->removeItem(selectedItem);
      Q_EMIT itemDeleted(selectedItem);
      delete selectedItem;
    }
    break;
  default:
    QGraphicsView::keyPressEvent(event);
  }
}

void Workbench::mouseMoveEvent(QMouseEvent* event) {
  if (currentCable) {
    currentCable->setDragPoint(mapToScene(event->pos()));
  }
  QGraphicsView::mouseMoveEvent(event);
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
