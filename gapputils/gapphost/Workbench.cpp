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

Workbench::Workbench(QWidget *parent) : QGraphicsView(parent), selectedItem(0),
    modifiable(true)
{
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

void Workbench::setModifiable(bool modifiable) {
  this->modifiable = modifiable;
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

vector<CableItem*>& Workbench::getCurrentCables() {
  return currentCables;
}

void Workbench::notifyItemChange(ToolItem* item) {
  Q_EMIT itemChanged(item);
}

void Workbench::mousePressEvent(QMouseEvent* event) {
  vector<ToolConnection*> connections;

  if (!modifiable) {
    QGraphicsView::mousePressEvent(event);
    return;
  }

  Q_FOREACH(QGraphicsItem* item, scene()->items()) {
    ToolItem* tool = dynamic_cast<ToolItem*>(item);
    if (tool) {
      QPointF mousePos = mapToScene(event->pos());
      int ex = mousePos.x();
      int ey = mousePos.y();
      int tx = item->x();
      int ty = item->y();
      connections.clear();
      if (tool->hitConnections(connections, ex - tx, ey - ty, ToolConnection::Output)) {
        // all connected cables are current cables
        for (unsigned i = 0; i < connections.size(); ++i) {
          ToolConnection* connection = connections[i];
          if (connection->cable) {
            CableItem* currentCable = connection->cable;
            currentCables.push_back(currentCable);
            currentCable->disconnectInput();
            currentCable->setDragPoint(mapToScene(event->pos()));
            Q_EMIT cableDeleted(currentCable);
          }
        }
        if (currentCables.size() == 0) {
          CableItem* currentCable = new CableItem(this, connections[0]);
          currentCables.push_back(currentCable);
          currentCable->setDragPoint(mapToScene(event->pos()));
          scene()->addItem(currentCable);
        }
        Q_FOREACH (QGraphicsItem *item, scene()->items()) {
          item->update();
        }
      }
      ToolConnection* connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Input);
      if (connection) {
        if (connection->cable) {
          CableItem* currentCable = connection->cable;
          currentCables.push_back(currentCable);
          currentCable->disconnectOutput();
          currentCable->setDragPoint(mapToScene(event->pos()));
          Q_EMIT cableDeleted(currentCable);
        } else {
          CableItem* currentCable = new CableItem(this, 0, connection);
          currentCables.push_back(currentCable);
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
  if (!modifiable) {
    QGraphicsView::mouseReleaseEvent(event);
    return;
  }

  if (currentCables.size()) {
    bool foundConnection = false;
    Q_FOREACH(QGraphicsItem* item, scene()->items()) {
      ToolItem* tool = dynamic_cast<ToolItem*>(item);
      if (tool) {
        QPointF mousePos = mapToScene(event->pos());
        int ex = mousePos.x();
        int ey = mousePos.y();
        int tx = item->x();
        int ty = item->y();
        if (currentCables.size() == 1 && currentCables[0]->needOutput()) {
          ToolConnection* connection = tool->hitConnection(ex - tx, ey - ty, ToolConnection::Input);
          if (connection && connection->property->getType() == currentCables[0]->getInput()->property->getType()) {
            foundConnection = true;
            currentCables[0]->setOutput(connection);
            currentCables[0]->endDrag();
            Q_EMIT cableCreated(currentCables[0]);
            break;
          }
        } else if (currentCables[0]->needInput()) {

          // Single Cable Case
          if (currentCables.size() == 1) {
            vector<ToolConnection*> connections;
            tool->hitConnections(connections, ex - tx, ey - ty, ToolConnection::Output);
            if (connections.size()) {
              ToolConnection* connection = connections[connections.size()-1];
              if (connection->property->getType() == currentCables[0]->getOutput()->property->getType()) {
                foundConnection = true;
                currentCables[0]->setInput(connection);
                currentCables[0]->endDrag();
                Q_EMIT cableCreated(currentCables[0]);
                break;
              }
            }
          }

          //TODO: Handle moving a bundle of connections.
        }
      }
    }
    if (!foundConnection) {
      for (unsigned i = 0; i < currentCables.size(); ++i) {
        scene()->removeItem(currentCables[i]);
        delete currentCables[i];
      }
    }
    currentCables.clear();
    Q_FOREACH (QGraphicsItem *item, scene()->items()) {
      item->update();
    }
  }
  QGraphicsView::mouseReleaseEvent(event);
}

void Workbench::removeToolItem(ToolItem* item) {
  scene()->removeItem(item);
  Q_EMIT itemDeleted(item);
  delete item;
}

void Workbench::keyPressEvent(QKeyEvent *event)
{
  if (!modifiable) {
    QGraphicsView::keyPressEvent(event);
    return;
  }

  switch (event->key()) {
  case Qt::Key_Delete:
    if (selectedItem && selectedItem->isDeletable()) {
      removeToolItem(selectedItem);
    }
    break;
  default:
    QGraphicsView::keyPressEvent(event);
  }
}

void Workbench::mouseMoveEvent(QMouseEvent* event) {
  for (unsigned i = 0; i < currentCables.size(); ++i)
    currentCables[i]->setDragPoint(mapToScene(event->pos()));
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
