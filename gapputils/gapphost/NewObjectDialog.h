#ifndef NEWOBJECTDIALOG_H
#define NEWOBJECTDIALOG_H

#include <QDialog>
#include "GeneratedFiles/ui_NewObjectDialog.h"

class NewObjectDialog : public QDialog
{
    Q_OBJECT

public:
    NewObjectDialog(QWidget *parent = 0);
    ~NewObjectDialog();

    QString getSelectedClass() const;
    void updateList();

private:
    Ui::NewObjectDialog ui;

private Q_SLOTS:
  void cancelButtonClicked(bool);
  void addButtonClicked(bool);
  void doubleClickedHandler(QListWidgetItem* item);
};

#endif // NEWOBJECTDIALOG_H
