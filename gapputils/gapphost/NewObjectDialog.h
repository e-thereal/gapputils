#ifndef NEWOBJECTDIALOG_H
#define NEWOBJECTDIALOG_H

#include <QDialog>
#include "ui_NewObjectDialog.h"

class NewObjectDialog : public QDialog
{
    Q_OBJECT

public:
    NewObjectDialog(QWidget *parent = 0);
    ~NewObjectDialog();

    QString getSelectedClass() const;

private:
    Ui::NewObjectDialog ui;

private Q_SLOTS:
  void cancelButtonClicked(bool);
  void addButtonClicked(bool);
};

#endif // NEWOBJECTDIALOG_H
