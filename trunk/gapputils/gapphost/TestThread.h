#pragma once

#include <qthread.h>

class TestThread : public QThread
{
  Q_OBJECT

public:
  virtual void run();

Q_SIGNALS:
  void progressed();
};

class TestResponder : public QThread {
  Q_OBJECT

public Q_SLOTS:
  void showProgress();
};
