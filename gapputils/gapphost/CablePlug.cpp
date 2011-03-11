#include "CablePlug.h"

#include <qpainter.h>

namespace gapputils {

CablePlug::CablePlug(QGraphicsItem* parent) : QGraphicsObject(parent)
{
  setFlag(ItemIsMovable);
}


CablePlug::~CablePlug(void)
{
}

QRectF CablePlug::boundingRect() const {
  return QRectF(0, 0, 10, 10);
}

void CablePlug::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  painter->setBrush(Qt::black);
  painter->drawRect(0, 0, 10, 10);
}

}
