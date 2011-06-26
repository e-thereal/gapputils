#include "GridDialog.h"

namespace gapputils {

namespace cv {

GridDialog::GridDialog(GridModel* model, int width, int height) : QDialog()
{
  setGeometry(50, 50, width + 50, height + 50);
  gridWidget = new GridWidget(model, width, height, this);
}


GridDialog::~GridDialog(void)
{
}

void GridDialog::renewGrid(int rowCount, int columnCount) {
  gridWidget->renewGrid(rowCount, columnCount);
}

void GridDialog::updateSize(int width, int height) {
  gridWidget->updateSize(width, height);
}

void GridDialog::setBackgroundImage(QImage* image) {
  gridWidget->setBackgroundImage(image);
}

void GridDialog::resizeEvent(QResizeEvent* resizeEvent) {
  gridWidget->setGeometry(0, 0, width(), height());
}

}

}
