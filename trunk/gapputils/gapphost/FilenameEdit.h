#ifndef FILENAMEEDIT_H
#define FILENAMEEDIT_H

#include <QFrame>
#include <qlineedit.h>
#include <qpushbutton.h>

class FilenameEdit : public QFrame
{
  Q_OBJECT
private:
  QLineEdit* edit;
  QPushButton* button;

public:
    FilenameEdit(QWidget *parent);
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
