#include "ShowImageDialog.h"

#include <QVBoxLayout>

ShowImageDialog::ShowImageDialog(QWidget *parent)
    : QDialog(parent)
{
  
  QVBoxLayout* layout = new QVBoxLayout(this);
  label = new QLabel();
  label->setScaledContents(true);
  layout->addWidget(label);
  this->setLayout(layout);
}

ShowImageDialog::~ShowImageDialog()
{
  delete label;
}

void ShowImageDialog::setImage(QImage* image) {
  label->setPixmap(QPixmap::fromImage(*image));
}
