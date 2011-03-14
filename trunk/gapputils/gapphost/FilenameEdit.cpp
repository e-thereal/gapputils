#include "FilenameEdit.h"

#include <qevent.h>
#include <qfiledialog.h>

FilenameEdit::FilenameEdit(QWidget *parent)
  : QFrame(parent)
{
  edit = new QLineEdit(this);
  edit->setVisible(true);
  edit->setFrame(false);
  button = new QPushButton("...", this);
  button->setVisible(true);
  this->setAutoFillBackground(true);
  this->setFrameStyle(QFrame::Box | QFrame::Plain);
  connect(edit, SIGNAL(editingFinished()), this, SLOT(editingFinishedHandler()));
  connect(button, SIGNAL(clicked(bool)), this, SLOT(clickedHandler()));
}

FilenameEdit::~FilenameEdit()
{
  delete edit;
}

void FilenameEdit::focusInEvent(QFocusEvent* e) {
  edit->setFocus();
}

void FilenameEdit::clickedHandler() {
  QFileDialog fileDialog(this);
  if (fileDialog.exec() == QDialog::Accepted) {
    QStringList filenames = fileDialog.selectedFiles();
    if (filenames.size()) {
      edit->setText(filenames[0]);
      Q_EMIT editingFinished();
    }
  }
}

void FilenameEdit::resizeEvent(QResizeEvent* event) {
  edit->setGeometry(1, 1, this->width()-27, this->height()-2);
  button->setGeometry(this->width()-27, 1, 26, this->height()-2);
}

void FilenameEdit::setText(const QString& text) {
  edit->setText(text);
}

QString FilenameEdit::getText() const {
  return edit->text();
}

void FilenameEdit::editingFinishedHandler() {
  if (!button->hasFocus())
    Q_EMIT editingFinished();
}
