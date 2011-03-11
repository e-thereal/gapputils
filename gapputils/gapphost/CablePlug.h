#pragma once
#ifndef _CABLEPLUG_H_
#define _CABLEPLUG_H_

#include <QGraphicsObject>

namespace gapputils {

class CablePlug : public QGraphicsObject
{
  Q_OBJECT

public:
  CablePlug(QGraphicsItem* parent = 0);
  virtual ~CablePlug(void);

  QRectF boundingRect() const;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

}

#endif
