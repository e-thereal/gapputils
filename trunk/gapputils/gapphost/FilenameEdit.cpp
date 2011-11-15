#include "FilenameEdit.h"

#define BOOST_FILESYSTEM_VERSION 2

#include <qevent.h>
#include <qfiledialog.h>
#include <boost/filesystem.hpp>

using namespace boost::filesystem;
using namespace capputils::attributes;

path makeRelative(const path& absolute) {
  path current = current_path();

  path::iterator ci = current.begin();
  path::iterator ai = absolute.begin();

  path relative;

  // skip what is the same
  for(; !ci->compare(*ai); ++ci, ++ai);
  for(; ci != current.end() && ci->compare("."); ++ci)
    relative /= "..";
  for(; ai != absolute.end(); ++ai)
    relative /= *ai;

  return relative;
}

bool inCurrentDir(const path filename) {
  path current = current_path();

  path::iterator ci = current.begin();
  path::iterator ai = filename.begin();

  for(; !ci->compare(*ai); ++ci, ++ai);

  return ci == current.end();
}

FilenameEdit::FilenameEdit(bool exists, FilenameAttribute* filenameAttribute, QWidget *parent)
  : exists(exists), filenameAttribute(filenameAttribute), QFrame(parent)
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
  QStringList filenames;
  bool multiSelection = filenameAttribute->getMultipleSelection();

  if (exists) {
    if (multiSelection)
      filenames = QFileDialog::getOpenFileNames(this, "Open Files", "", filenameAttribute->getPattern().c_str());
    else {
      QString filename = QFileDialog::getOpenFileName(this, "Open File", "", filenameAttribute->getPattern().c_str());
      if (!filename.isNull())
        filenames.append(filename);
    }
  } else {
    QString filename = QFileDialog::getSaveFileName(this, "Save File", "", filenameAttribute->getPattern().c_str());
    if (!filename.isNull())
      filenames.append(filename);
  }
  if (filenames.size()) {
    if (multiSelection) {
      QString filenamesString;
      for (int i = 0; i < filenames.count(); ++i) {
        path filename(filenames[i].toAscii().data());
        if (inCurrentDir(filename))
          filename = makeRelative(filename);
        if (i > 0)
          filenamesString += " ";
        filenamesString += QString("\"") + filename.file_string().c_str() + "\"";
      }
      edit->setText(filenamesString);
    } else {
      path filename(filenames[0].toAscii().data());
      if (inCurrentDir(filename))
        filename = makeRelative(filename);
      edit->setText(filename.file_string().c_str());
    }
    Q_EMIT editingFinished();
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
