#ifndef SHOWIMAGEDIALOG_H
#define SHOWIMAGEDIALOG_H

#include <QDialog>
#include <QImage>
#include <qlabel.h>

class ShowImageDialog : public QDialog
{
  Q_OBJECT

private:
  QLabel* label;

public:
  ShowImageDialog(QWidget *parent = 0);
  virtual ~ShowImageDialog();

  void setImage(QImage* image);
};

#endif // SHOWIMAGEDIALOG_H
