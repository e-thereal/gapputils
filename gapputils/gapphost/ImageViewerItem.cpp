#include "ImageViewerItem.h"

#include <qpainter.h>
#include "ImageViewer.h"

using namespace capputils::reflection;

namespace gapputils {

using namespace workflow;

ImageViewerItem::ImageViewerItem(Node* node, Workbench *bench) : ToolItem(node, bench)
{
  width = 100;
  height = 70;
  updateConnectionPositions();
}

ImageViewerItem::~ImageViewerItem(void)
{
}

void ImageViewerItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  drawBox(painter);
  drawConnections(painter, false);
  QFont font = painter->font();
  //font.setBold(true);
  //font.setItalic(true);
  font.setPointSize(10);
  painter->setFont(font);
  painter->drawText(0, 0, width, height, Qt::AlignCenter, QString("Double click\nto view image.\n(") + QString(getLabel().c_str()) + ")");
}

void ImageViewerItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  ImageViewer* viewer = dynamic_cast<ImageViewer*>(getNode()->getModule());
  if (viewer) {
    viewer->showImage();
  }
}
  
}
