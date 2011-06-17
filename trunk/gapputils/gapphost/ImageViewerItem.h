#pragma once
#ifndef _IMAGEVIEWERITEM_H_
#define _IMAGEVIEWERITEM_H_

#include "ToolItem.h"

namespace gapputils {

class ImageViewerItem : public ToolItem
{
public:
  ImageViewerItem(const std::string& label, Workbench *bench = 0);
  virtual ~ImageViewerItem(void);

  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
  virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);
};

}

#endif
