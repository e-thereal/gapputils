#ifndef FILENAMEEDIT_H
#define FILENAMEEDIT_H

#include <QFrame>
#include <qlineedit.h>
#include <qpushbutton.h>

#include <capputils/FilenameAttribute.h>

class FilenameEdit : public QFrame
{
  Q_OBJECT
private:
  QLineEdit* edit;
  QPushButton* button;
  bool exists;
  capputils::attributes::FilenameAttribute* filenameAttribute;

public:
    FilenameEdit(bool exists, capputils::attributes::FilenameAttribute* filenameAttribute,
        QWidget *parent);
    ~FilenameEdit();

    void setText(const QString& text);
    QString getText() const;

protected:
  virtual void resizeEvent(QResizeEvent* event);
  virtual void focusInEvent(QFocusEvent* e);

protected Q_SLOTS:
  void editingFinishedHandler();
  void clickedHandler();
  

Q_SIGNALS:
  void editingFinished();
};

#endif // FILENAMEEDIT_H
